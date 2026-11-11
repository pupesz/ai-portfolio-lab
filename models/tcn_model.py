import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.dropout(F.relu(self.chomp1(self.conv1(x))))
        y = self.dropout(F.relu(self.chomp2(self.conv2(y))))
        return F.relu(y + self.downsample(x))

class TCN(nn.Module):
    def __init__(self, n_features, channels=64, levels=5, kernel_size=7, dropout=0.1, out_dim=1,
                 head_norm=False):
        super().__init__()
        layers = []
        in_ch = n_features
        for i in range(levels):
            layers.append(TemporalBlock(in_ch, channels, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = channels
        self.net = nn.Sequential(*layers)
        # LayerNorm before the head stabilises regression on a zero-centred
        # target (the final TemporalBlock ends in ReLU, so without this the
        # head sees only non-negative, large-magnitude activations and the
        # predictions overshoot). Off by default to preserve original behaviour.
        self.norm = nn.LayerNorm(channels) if head_norm else nn.Identity()
        self.head = nn.Linear(channels, out_dim)

    def forward(self, x):                 # x: (batch, time, features)
        x = x.transpose(1, 2)             # -> (batch, features, time)
        y = self.net(x)[:, :, -1]         # last timestep
        return self.head(self.norm(y))
