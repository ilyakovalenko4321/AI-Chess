import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()

        # CNN блок: вход 12 каналов (6 фигур × 2 цвета), доска 8×8
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Policy head (куда ходить)
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4672)  # все возможные ходы

        # Value head (оценка позиции)
        self.value_conv = nn.Conv2d(128, 16, kernel_size=1)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # CNN feature extractor
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # значение от -1 до 1

        return p, v
