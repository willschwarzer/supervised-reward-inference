import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=151):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                  math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x

class MLP(nn.Module):
    def __init__(self, demonstration_rep_dim, state_rep_dim, hidden_size, out_dim):
        super().__init__()
        combined_dim = demonstration_rep_dim + state_rep_dim
        self.linear1 = nn.Linear(combined_dim, hidden_size)
        self.relu = nn.LeakyReLU()
        # self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2 = nn.Linear(hidden_size, out_dim)

    def forward(self, demonstration_rep, state_rep):
        combined_rep = torch.cat((demonstration_rep, state_rep), axis=-1)
        x = self.linear1(combined_rep)
        x = self.relu(x)
        x = self.linear2(x)
        return x.squeeze()


class StateNet(nn.Module):
    def __init__(self, state_rep_dim, hidden_size, num_layers, obs_size):
        super().__init__()
        modules = [nn.Linear(obs_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU()]
        for _ in range(num_layers-1):
            modules += [nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU()]
        modules += [nn.Linear(hidden_size, state_rep_dim)]
        self.net = nn.Sequential(*modules)

    def forward(self, state):
        return self.net(state)


class DemonstrationNetTransformer(nn.Module):
    def __init__(
            self,
            obs_size,
            horizon,
            num_transformer_layers,
            demonstration_hidden_size,
            demonstration_rep_dim,
            sigmoid=False,
            nhead=None):
        super().__init__()
        self.positional_encoder = PositionalEncoder(obs_size, max_seq_len=horizon+1) # not sure why this is +1 but we'll go with what we had
        if nhead is None:
            # Preserve historical behavior when possible, but gracefully handle
            # new observation sizes (e.g., policy features with dim=8).
            if obs_size % 3 == 0:
                nhead = 3
            elif obs_size % 2 == 0:
                nhead = 2
            else:
                nhead = 1
        if obs_size % nhead != 0:
            raise ValueError(f"obs_size={obs_size} must be divisible by nhead={nhead}")
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=obs_size,
            nhead=nhead,
            dim_feedforward=demonstration_hidden_size,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_transformer_layers)
        self.linear1 = nn.Linear(obs_size, 50) # not sure why we hard-coded 50 here ¯\_(ツ)_/¯
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(50, demonstration_rep_dim)
        self.sigmoid = nn.Sigmoid() if sigmoid else nn.Identity()
        self.obs_size = obs_size
        self.horizon = horizon
        self.nhead = nhead

    def forward(self, states):
        # states: (batch size, (n,) horizon, obs dim)
        if len(states.shape) == 4:
            n = states.shape[1]
            states = states.reshape(-1, states.shape[-2], states.shape[-1])
        else:
            n = None
        x = self.positional_encoder(states)
        x = self.transformer(x)
        assert(x.shape[1] == self.horizon)
        assert(x.shape[2] == self.obs_size)
        x = torch.mean(x, dim=1)
        if n is not None:
            x = x.reshape(-1, n, x.shape[-1])
            x = torch.sum(x, dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        #x = x.view(-1, 3)
        x = self.sigmoid(x)
        return x

class DemonstrationNetLSTM(nn.Module):
    def __init__(self, obs_size, num_lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(25*obs_size, 2048, num_lstm_layers, batch_first=True)
        self.linear1 = nn.Linear(2048, 16)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(16, 3)

    def forward(self, states):
        x, _ = self.lstm(states)
        x = x[:, -1, :]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TST(nn.Module):
    # TST: Transformer->Set Transformer
    def __init__(self, demonstration_transformer, set_transformer):
        super().__init__()
        self.demonstration_transformer = demonstration_transformer
        self.set_transformer = set_transformer

    def forward(self, demonstrations):
        # demonstrations: (bsize, n, L, |S|)
        # flatten to (bsize*n, L, |S|)
        demonstrations_flattened = demonstrations.view(
            -1, demonstrations.shape[-2], demonstrations.shape[-1])
        # pass through demonstration transformer
        dem_reps = self.demonstration_transformer(demonstrations_flattened)
        # dem_reps: (bsize*n, D) for some D
        # reshape to (bsize, n, D)
        dem_reps = dem_reps.view(-1, demonstrations.shape[1],
                                 dem_reps.shape[-1])
        # pass through set transformer
        set_reps = self.set_transformer(dem_reps)  # (bsize, dem_rep_dim)
        return set_reps
