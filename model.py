import torch


class MLP(torch.nn.Module):
    def __init__(self, device, n_layers, input_dim, hidden_size, output_size):
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.ln0 = torch.nn.LayerNorm(hidden_size).to(device)
        self.lns = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_size) for _ in range(n_layers - 2)
        ]).to(device)
        self.lnf = torch.nn.LayerNorm(output_size).to(device)
        self.W0 = torch.nn.Linear(input_dim, hidden_size).to(device)
        self.Ws = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size) for _ in range(n_layers - 2)
        ]).to(device)
        self.Wf = torch.nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x):
        x = self.ln0(self.dropout(torch.relu(self.W0(x))))
        for w, ln in zip(self.Ws, self.lns):
            x = ln(x + self.dropout(torch.relu(w(x))))
        x = self.lnf(self.Wf(x))
        return x


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
        self.phi = MLP(device, 2, decoder_features, context_dims, context_dims)
        self.psi = MLP(device, 2, listener_features, context_dims, context_dims)

    def forward(self, s, h):
        s = self.phi(s)
        h = self.psi(h)
        e = torch.einsum('bd,brd->br', s, h)
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
        self.hidden_size = hidden_size
        self.arange = torch.arange(vocab_size, device=device)
        self.output = MLP(device, 3, hidden_size, hidden_size, vocab_size)

    def forward(self, h, y, sample_prob=0.1):
        batch_size = h.shape[0]
        # initial states
        s0 = torch.zeros((batch_size, self.hidden_size), device=self.device)
        s1 = torch.zeros((batch_size, self.hidden_size), device=self.device)

        # initial cell states
        cs0 = torch.zeros((batch_size, self.hidden_size), device=self.device)
        cs1 = torch.zeros((batch_size, self.hidden_size), device=self.device)

        # sampling probability
        prob = torch.empty((batch_size,), device=self.device)
        prob.fill_(sample_prob)

        # rnn
        output = []
        for i in range(y.shape[1]):
            if i == 0:
                z = y[:, i]
            else:
                # sample from previous time step
                pr = torch.softmax(output[-1], dim=1).detach()
                sampled = torch.multinomial(pr, 1).reshape(batch_size)
                # choose between target and sampled outputs
                do_sample = torch.bernoulli(prob).byte()
                z = torch.where(do_sample, sampled, y[:, i])
            # one-hot encoding
            z = (z.unsqueeze(1) == self.arange.unsqueeze(0)).float()
            # attention
            c = self.attn(s1, h)
            # two-layer lstm
            inputs = torch.cat([z, c], dim=1)
            s0, cs0 = self.lstm0(inputs, (s0, cs0))
            s1, cs1 = self.lstm1(s0, (s1, cs1))
            # project to output space
            o = self.output(s1)
            output.append(o)
        output = torch.stack(output, dim=1)

        return output


class LAS(torch.nn.Module):
    def __init__(self, device, vocab_size, pad):
        super(LAS, self).__init__()
        self.listener = Listener(device)
        self.attend_spell = AttendAndSpell(device, vocab_size)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.0)
        self.step = 1
        self.device = device

    def forward(self, source, target):
        h = self.listener(source)
        o = self.attend_spell(h, target)
        return o

    def loss(self, source, target):
        o = self.forward(source, target[:, :-1])
        loss = self.loss_func(o.reshape(-1, o.shape[-1]), target[:, 1:].reshape(-1))
        return loss

    def train_step(self, source, target):
        loss = self.loss(source, target)
        lr = 0.002 * (0.98 ** (self.step // 100))
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

    def predict(self, source):
        h = self.listener(source)
