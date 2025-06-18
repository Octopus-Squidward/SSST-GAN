import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from .SpatVit import SpatViT
from .SpecVit import SpecViT


class SSST_GAN(torch.nn.Module):
    def __init__(self, patch_size, channels, seg_patches, NUM_TOKENS, embed_dim, num_em):
        super(SSST_GAN, self).__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.NUM_TOKENS = NUM_TOKENS
        self.seg_patches = seg_patches
        self.embed_dim = embed_dim
        self.num_em = num_em

        self.spat_encoder = SpatViT(img_size=self.patch_size,
                                    in_chans=self.channels,
                                    use_checkpoint=True,
                                    patch_size=self.seg_patches,
                                    drop_path_rate=0.1, out_indices=[3, 5, 7, 11], embed_dim=768,
                                    depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop_rate=0., attn_drop_rate=0., use_abs_pos_emb=False, n_points=8)

        self.spec_encoder = SpecViT(
            NUM_TOKENS=self.NUM_TOKENS,
            img_size=self.patch_size,
            in_chans=self.channels,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=False,
            use_abs_pos_emb=False,
            interval=3
        )

        self.conv_features1 = nn.Conv2d(self.embed_dim, self.NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec1 = nn.Sequential(
            nn.Linear(self.NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.conv_features2 = nn.Conv2d(self.embed_dim, self.NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec2 = nn.Sequential(
            nn.Linear(self.NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.conv_features3 = nn.Conv2d(self.embed_dim, self.NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec3 = nn.Sequential(
            nn.Linear(self.NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.conv_features4 = nn.Conv2d(self.embed_dim, self.NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec4 = nn.Sequential(
            nn.Linear(self.NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.NUM_TOKENS, self.NUM_TOKENS, kernel_size=1),
            nn.LeakyReLU(0.02),  # nn.ReLU(),
            nn.BatchNorm2d(self.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.NUM_TOKENS, self.NUM_TOKENS, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.NUM_TOKENS, self.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.NUM_TOKENS, self.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.NUM_TOKENS),
            nn.Dropout(0.2),
        )

        self.conv2_ = nn.Sequential(
            nn.Conv2d(self.NUM_TOKENS * 2, self.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv3_ = nn.Sequential(
            nn.Conv2d(self.NUM_TOKENS * 2, self.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv4_ = nn.Sequential(
            nn.Conv2d(self.NUM_TOKENS * 2, self.NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.NUM_TOKENS),
            nn.Dropout(0.2),
        )

        self.smooth = nn.Sequential(
            nn.Conv2d(self.NUM_TOKENS * 4, self.NUM_TOKENS * 2, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.NUM_TOKENS * 2),
            # nn.Dropout(0.2),
            nn.Conv2d(self.NUM_TOKENS * 2, self.NUM_TOKENS, kernel_size=(1, 1)))

        self.conv5 = nn.Sequential(
            nn.Conv2d(self.NUM_TOKENS, self.num_em, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(self.num_em),
            # nn.Dropout(0.2),
        )

        self.decoder = nn.Conv2d(in_channels=self.num_em,
                                 out_channels=self.channels,
                                 kernel_size=1, stride=1, bias=False)

        self.decoder1 = nn.Conv2d(in_channels=self.channels,
                                  out_channels=self.channels,
                                  kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()

    def forward_fusion(self, x):
        b, _, h, w = x.shape

        img_features = self.spat_encoder(x)

        img_fea = []
        ops = [self.conv_features1, self.conv_features2, self.conv_features3, self.conv_features4]

        for i in range(len(ops)):
            img_fea.append(ops[i](img_features[i + 1]))

        spec_features = self.spec_encoder(x)
        spec_feature = spec_features[-1]
        spec_feature = self.pool(spec_feature).view(b, -1)

        spec_weights = []
        ops_ = [self.fc_spec1, self.fc_spec2, self.fc_spec3, self.fc_spec4]
        for i in range(len(ops_)):
            spec_weights.append((ops_[i](spec_feature)).view(b, -1, 1, 1))
        ss_feature = []
        ss_feature.append(x)
        for i in range(4):
            ss_feature.append((1 + spec_weights[i]) * img_fea[i])
        return ss_feature

    def getAbundances(self, data):
        H, W = data.shape[2], data.shape[3]
        x = self.forward_fusion(data)
        p4 = self.conv1(x[4])
        p3 = self.conv2(x[3])
        p2 = self.conv3(x[2])
        p1 = self.conv4(x[1])
        p1 = torch.cat([p1, p2, p3, p4], dim=1)

        p1 = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=True)
        p1 = self.smooth(p1)
        x = self.conv5(p1)
        abunds = F.softmax(x, dim=1)
        abunds = abunds
        return abunds

    def forward(self, inputs):
        abunds = self.getAbundances(inputs)

        output_l = self.decoder(abunds)
        out_nl = self.relu(self.decoder1(output_l))
        out_nl = self.relu(self.decoder1(out_nl))
        out_nl = self.relu(self.decoder1(out_nl))
        output = output_l + out_nl

        endmembers = self.decoder.weight.data
        endmembers = np.squeeze(endmembers)
        return abunds, output, endmembers, out_nl
