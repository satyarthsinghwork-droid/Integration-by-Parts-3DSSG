import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Point-MAE Encoder

    Input:
        (B, G, N, 3)

    Output:
        (B, G, encoder_channel)
    """

    def __init__(self, encoder_channel=384):
        super().__init__()

        self.encoder_channel = encoder_channel

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1)
        )

    def forward(self, point_groups):

        """
        point_groups

        Shape:
            B × G × N × 3

        Example:
            B=1
            G=64
            N=32
        """

        B, G, N, _ = point_groups.shape

        point_groups = point_groups.reshape(B * G, N, 3)

        feature = self.first_conv(
            point_groups.transpose(2, 1)
        )

        feature_global = torch.max(
            feature,
            dim=2,
            keepdim=True
        )[0]

        feature = torch.cat(
            [
                feature_global.expand(-1, -1, N),
                feature
            ],
            dim=1
        )

        feature = self.second_conv(feature)

        feature_global = torch.max(
            feature,
            dim=2
        )[0]

        feature_global = feature_global.reshape(
            B,
            G,
            self.encoder_channel
        )

        return feature_global