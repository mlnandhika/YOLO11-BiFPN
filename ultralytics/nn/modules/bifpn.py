import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)
    
class WeightedAdd(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))
        self.eps = 1e-4

    def forward(self, inputs):
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)
        return sum(w[i] * inputs[i] for i in range(len(inputs)))

class BiFPNLayer(nn.Module):
    def __init__(self, ch):
        super().__init__()

        # weighted fusion
        self.w1 = WeightedAdd(2)
        self.w2 = WeightedAdd(2)

        # conv biasa
        self.conv3 = ConvBNAct(ch, ch)
        self.conv4 = ConvBNAct(ch, ch)
        self.conv5 = ConvBNAct(ch, ch)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.down = nn.MaxPool2d(2)

    def forward(self, p3, p4, p5):
        # ===== TOP DOWN PATH =====
        p4_td = self.w1([p4, self.up(p5)])
        p3_td = self.w1([p3, self.up(p4_td)])

        # ===== BOTTOM UP PATH =====
        p4_out = self.w2([p4_td, self.down(p3_td)])
        p5_out = self.w2([p5, self.down(p4_out)])

        # conv smoothing
        p3_out = self.conv3(p3_td)
        p4_out = self.conv4(p4_out)
        p5_out = self.conv5(p5_out)

        return p3_out, p4_out, p5_out

class BiFPN(nn.Module):
    def __init__(self, ch, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [BiFPNLayer(ch) for _ in range(n_layers)]
        )

    def forward(self, p3, p4, p5):
        for layer in self.layers:
            p3, p4, p5 = layer(p3, p4, p5)
        return p3, p4, p5