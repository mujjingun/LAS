import torch
import math
import tqdm


class Listener(torch.nn.Module):
    def __init__(self, device, input_dim=40, hidden_size=256):
        super(Listener, self).__init__()
        self.lstms = torch.nn.ModuleList([
            torch.nn.LSTM(input_dim, hidden_size // 2, bidirectional=True).to(device),
            torch.nn.LSTM(hidden_size * 2, hidden_size // 2, bidirectional=True).to(device),
            torch.nn.LSTM(hidden_size * 2, hidden_size // 2, bidirectional=True).to(device)
        ])

    def forward(self, x):
        x = x.transpose(0, 1)
        for lstm in self.lstms:
            x, _ = lstm(x)
            x = torch.cat([x[0::2], x[1::2]], dim=2)
        x = x.transpose(0, 1)
        return x


class AttentionContext(torch.nn.Module):
    def __init__(self, device, listener_features, decoder_features, context_dims):
        super(AttentionContext, self).__init__()
        self.context_dims = context_dims
        self.phi = torch.nn.Linear(decoder_features, context_dims).to(device)
        self.psi = torch.nn.Linear(listener_features, context_dims).to(device)

    def forward(self, s, h):
        s = self.phi(s)
        h = self.psi(h)
        e = torch.einsum('bd,brd->br', s, h) / math.sqrt(self.context_dims)
        alpha = torch.softmax(e, dim=1)
        c = torch.einsum('br,brd->bd', alpha, h)
        return c


class AttendAndSpell(torch.nn.Module):
    def __init__(self, device, vocab_size, hidden_size=512):
        super(AttendAndSpell, self).__init__()
        self.attn = AttentionContext(device, hidden_size, hidden_size, hidden_size)
        self.lstm0 = torch.nn.LSTMCell(hidden_size + vocab_size, hidden_size).to(device)
        self.lstm1 = torch.nn.LSTMCell(hidden_size, hidden_size).to(device)
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.arange = torch.arange(vocab_size, device=device)
        self.output = torch.nn.Linear(hidden_size * 2, vocab_size).to(device)

    def initial_states(self, batch_size):
        # initial states
        s0 = torch.zeros((batch_size, self.hidden_size), device=self.device)
        s1 = torch.zeros((batch_size, self.hidden_size), device=self.device)

        # initial cell states
        cs0 = torch.zeros((batch_size, self.hidden_size), device=self.device)
        cs1 = torch.zeros((batch_size, self.hidden_size), device=self.device)

        return (s0, cs0), (s1, cs1)

    def step(self, z, h, c, states):
        (s0, s1), (cs0, cs1) = states
        # one-hot encoding
        z = (z.unsqueeze(1) == self.arange.unsqueeze(0)).float()
        # two-layer lstm
        inputs = torch.cat([z, c], dim=1)
        s0, cs0 = self.lstm0(inputs, (s0, cs0))
        s1, cs1 = self.lstm1(s0, (s1, cs1))
        # attention
        c = self.attn(s1, h)
        # project to output space
        o = self.output(torch.cat([s1, c], dim=1))
        return o, c, ((s0, cs0), (s1, cs1))

    def forward(self, h, y, tf_rate):
        batch_size = h.shape[0]
        states = self.initial_states(batch_size)

        # sampling probability
        prob = torch.empty((batch_size,), device=self.device).fill_(tf_rate)

        # rnn
        output = []
        z = y[:, 0]
        c = self.attn(states[-1][0], h)
        for i in range(y.shape[1]):
            # advance one time step
            o, c, states = self.step(z, h, c, states)
            # append output
            output.append(o)
            # evaluate next feeback input
            if i + 1 < y.shape[1]:
                # sample from output
                pr = torch.softmax(o, dim=1)
                sampled = torch.multinomial(pr, 1).reshape(batch_size)
                # choose between target and sampled outputs
                do_sample = torch.bernoulli(prob).byte()
                z = torch.where(do_sample, sampled, y[:, i + 1]).detach()
        output = torch.stack(output, dim=1)
        return output

    def predict(self, h, max_length):
        batch_size = h.shape[0]
        states = self.initial_states(batch_size)

        # start with sos
        target = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        c = self.attn(states[-1][0], h)
        for length in range(1, max_length + 1):
            # advance one time step
            o, c, states = self.step(target[:, -1], h, c, states)
            # select greedily
            pr = torch.softmax(o, dim=1)
            new_col = torch.argmax(pr, dim=1).unsqueeze(1)
            target = torch.cat([target, new_col], dim=1)
        return target


class LAS(torch.nn.Module):
    def __init__(self, device, vocab_size, pad, start_lr, decay_steps):
        super(LAS, self).__init__()
        self.listener = Listener(device)
        self.attend_spell = AttendAndSpell(device, vocab_size)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.0)
        self.step = 1
        self.device = device
        self.start_lr = start_lr
        self.decay_steps = decay_steps

    def forward(self, source, target, tf_rate):
        h = self.listener(source)
        o = self.attend_spell(h, target, tf_rate)
        return o

    def loss(self, source, target, tf_rate):
        o = self.forward(source, target[:, :-1], tf_rate)
        loss = self.loss_func(o.reshape(-1, o.shape[-1]), target[:, 1:].reshape(-1))
        return loss

    def train_step(self, source, target):
        loss = self.loss(source, target, 0.1)
        lr = self.start_lr * (0.98 ** (self.step // self.decay_steps))
        for group in self.optim.param_groups:
            group['lr'] = lr
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optim.step()
        self.step += 1

        return loss.item()

    def save(self, file_name):
        save_state = {
            'step': self.step,
            'weights': self.state_dict(),
            'optim': self.optim.state_dict(),
        }
        torch.save(save_state, file_name)

    def load(self, file_name):
        load_state = torch.load(file_name, map_location=self.device)
        self.load_state_dict(load_state['weights'])
        self.optim.load_state_dict(load_state['optim'])
        self.step = load_state['step']

    def predict(self, source, max_length):
        with torch.no_grad():
            h = self.listener(source)
            pred = self.attend_spell.predict(h, max_length)
            return pred.cpu().numpy().tolist()
