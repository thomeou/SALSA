import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import ConvBlock, _ResNet, _ResnetBasicBlock


class BaseEncoder(nn.Module):
    def __init__(
        self,
        n_input_channels: int = 1,
        n_output_channels: int = 512,
        p_dropout: float = 0.0,
        time_downsample_ratio: int = 16,
        **kwargs
    ):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.p_dropout = p_dropout
        self.n_output_channels = n_output_channels
        self.time_downsample_ratio = time_downsample_ratio


class PannResNet22(BaseEncoder):
    """
    Derived from PANN ResNet22 network. PannResNet22L17 has 4 basic resnet blocks
    """

    def __init__(self, n_input_channels: int = 1, p_dropout: float = 0.0, **kwargs):
        """
        :param n_input_channels: Number of input channels.
        :param p_dropout: Dropout probability.
        :param pretrained: If True, load pretrained model.
        """
        super().__init__(
            n_input_channels=n_input_channels,
            n_output_channels=512,
            p_dropout=p_dropout,
            time_downsample_ratio=16,
        )

        self.conv_block1 = ConvBlock(in_channels=n_input_channels, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[2, 2, 2, 2], zero_init_residual=True)

    def forward(self, x):
        """
        Input: Input x: (batch_size, n_channels, n_timesteps, n_features)"""

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.p_dropout, training=self.training, inplace=True)
        x = self.resnet(x)

        return x


if __name__ == "__main__":
    encoder = PannResNet22(n_input_channels=7)
    pytorch_total_params = sum(
        p.numel() for p in encoder.parameters() if p.requires_grad
    )
    print("number of trainable params: {}".format(pytorch_total_params))
    # print(type(encoder))
    # print(encoder.__dict__)
    # print(encoder)

    x = torch.rand((16, 7, 320, 128))
    y = encoder.forward(x)
    print(y.shape)
    print('time downsample ratio: {}'.format(320 / y.shape[2]))
    print('freq downsample ratio: {}'.format(128 / y.shape[3]))
