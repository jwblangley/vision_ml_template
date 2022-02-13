import torch
import torch.nn as nn

from model.components.residual import ResidualBlock, ResidualBlockResize

def _lerp(a, b , t):
    return a + t * (b - a)

def _down_block(in_channels, out_channels, factor):
    middle_channels = (in_channels + out_channels) // 2

    return nn.Sequential(
        ResidualBlock(in_channels, middle_channels),
        ResidualBlockResize(middle_channels, middle_channels, factor=1 / factor),
        ResidualBlock(middle_channels, out_channels),
    )

def _up_block(in_channels, out_channels, factor):
    middle_channels = (in_channels + out_channels) // 2

    return nn.Sequential(
        ResidualBlock(in_channels, middle_channels),
        ResidualBlockResize(middle_channels, middle_channels, factor=factor),
        ResidualBlock(middle_channels, out_channels),
    )

class ResUNet(nn.Module):
    """
    Implementation of a Residual UNet
    """
    def __init__(self, in_channels, out_channels, max_channels=256, layers=5, factor=2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_channels = max_channels
        self.layers = layers
        self.factor = factor

        channel_t_pairs = list(zip(range(self.layers + 1), [*range(1, self.layers + 1), None]))[:-1]

        self.encoder_blocks = [
            _down_block(int(_lerp(self.in_channels, self.max_channels, t1 / self.layers)), int(_lerp(self.in_channels, self.max_channels, t2 / self.layers)), self.factor)
            for t1, t2 in channel_t_pairs
        ]

        # 2x so that skip connections can be added
        self.decoder_blocks = [
            _up_block(2 * int(_lerp(self.in_channels, self.max_channels, t1 / self.layers)), int(_lerp(self.in_channels, self.max_channels, t2 / self.layers)), self.factor)
            for t2, t1 in reversed(channel_t_pairs)
        ]

        self._embed_layers()

    """
    Required for these blocks to appear as parameters of this model
    """
    def _embed_layers(self):
        for i, enc_block in enumerate(self.encoder_blocks):
            setattr(self, f"enc_block_{i}", enc_block)

        for i, dec_block in enumerate(self.decoder_blocks):
            setattr(self, f"dec_block_{i}", dec_block)

    def forward(self, x):
        skip_cons = []

        prev = x
        for block in self.encoder_blocks:
            cur = block(prev)
            skip_cons.append(cur)

            prev = cur

        for i, block in enumerate(self.decoder_blocks):
            skip_con_vals = skip_cons[self.layers - 1 - i]
            combined = torch.cat([prev, skip_con_vals], 1)

            cur = block(combined)

            prev = cur

        return prev
