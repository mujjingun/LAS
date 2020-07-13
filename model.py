import torch


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
        self.phi = torch.nn.Linear(decoder_features, context_dims).to(device)
        self.psi = torch.nn.Linear(listener_features, context_dims).to(device)

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
        self.output = torch.nn.Linear(hidden_size, vocab_size).to(device)

    def forward(self, h, y):
        batch_size = h.shape[0]
        # initial states
        s0 = torch.zeros((batch_size, self.hidden_size), device=self.device)
        s1 = torch.zeros((batch_size, self.hidden_size), device=self.device)

        # initial cell states
        cs0 = torch.zeros((batch_size, self.hidden_size), device=self.device)
        cs1 = torch.zeros((batch_size, self.hidden_size), device=self.device)

        # to one hot encoding
        y = (y.unsqueeze(2) == self.arange.unsqueeze(0).unsqueeze(1)).float()

        # rnn
        s = []
        for i in range(y.shape[1]):
            c = self.attn(s1, h)
            inputs = torch.cat([y[:, i], c], dim=1)
            s0, cs0 = self.lstm0(inputs, (s0, cs0))
            s1, cs1 = self.lstm1(s0, (s1, cs1))
            s.append(s1)
        s = torch.stack(s, dim=1)

        # project to output space
        output = self.output(s)
        return output


class LAS(torch.nn.Module):
    def __init__(self, device, vocab_size, pad):
        super(LAS, self).__init__()
        self.listener = Listener(device)
        self.attend_spell = AttendAndSpell(device, vocab_size)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.2, betas=(0.9, 0.98), eps=10e-9)
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
        source, target = source.to(self.device), target.to(self.device)
        loss = self.loss(source, target)
        lr = 0.2 * (0.98 ** (self.step // 500))
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
