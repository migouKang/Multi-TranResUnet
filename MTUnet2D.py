import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from modules.abla_unet import CBB


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_size=768, img_size=512):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(self.n_patches, emb_size))

    def forward(self, x):
        b = x.shape[0]
        x = self.projection(x) 
        x = x.flatten(2).transpose(1, 2)
        x += self.positions
        return x


class ViTBlock(nn.Module):
    def __init__(self, emb_size=768, n_heads=12, mlp_dim=3072, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialMap(nn.Module):
    def __init__(self, in_ch, out_ch, spa_size, bn=False, relu=True, bias=True):
        super(SpatialMap, self).__init__()
        init_ch = (in_ch // 2) if in_ch > out_ch else (in_ch * 2)
        self.conv_fc = nn.Sequential(
            Conv(inp_dim=in_ch, out_dim=init_ch, bn=True),
            Conv(inp_dim=init_ch, out_dim=out_ch, bn=True)
        )
        self.spa_size = spa_size

    def forward(self, x):
        x = self.conv_fc(x) 
        x_up = F.interpolate(x, size=[self.spa_size, self.spa_size], mode='bilinear', align_corners=False)
        return x_up


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class CBAM(nn.Module):
    def __init__(self, in_ch, rate, only_ch=0):
        super(CBAM, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=in_ch, out_channels=int(in_ch / rate), kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels=int(in_ch / rate), out_channels=in_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        self.fc3 = nn.Conv2d(in_channels=in_ch, out_channels=int(in_ch / rate), kernel_size=1)
        self.fc4 = nn.Conv2d(in_channels=int(in_ch / rate), out_channels=in_ch, kernel_size=1)

        self.ch_use = only_ch

    def forward(self, x):
        x_in = x 
        x = x.mean((2, 3), keepdim=True)  
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x)  
        x = self.sigmoid(x) * x_in  
        if self.ch_use == 1:
            return x
        elif self.ch_use == 0:
            x = x

        s_in = x  
        s = self.compress(x)  
        s = self.spatial(s)  
        s = self.sigmoid(s) * s_in  

        return s


class HFS(nn.Module):
    def __init__(self, channels, res=0):
        super(HFS, self).__init__()
        self.in_channel_0 = channels[0] 
        self.in_channel_1 = channels[1] 
        self.in_channel_2 = channels[2] 
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels[0]+channels[1], out_channels=channels[2], kernel_size=3, bias=True),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
        )
        self.res_mode = res
        self.att = CBAM(in_ch=channels[2], rate=4)
        self.fuse_conv2 = Conv(inp_dim=channels[2]*2, out_dim=channels[2], bn=True)

    def forward(self, x):
        hi3 = x[0]
        hi6 = x[1]
        x_in = x[2]

        hi3_4 = F.interpolate(hi3, size=[4, 4], mode='bilinear', align_corners=False)
        hi6_4 = F.interpolate(hi6, size=[4, 4], mode='bilinear', align_corners=False)
        hi_3 = self.fuse_conv1(torch.cat([hi3_4, hi6_4], dim=1))
        hi_3 = F.interpolate(hi_3, size=[8, 8], mode='bilinear', align_corners=False)
        hi_3 = self.att(hi_3)
        x_out = self.fuse_conv2(torch.cat([hi_3, x_in], dim=1))
        if self.res_mode == 1:
            x_out = x_out + x_in
        return x_out


class ResUP(nn.Module):
    def __init__(self, channels, res=0):
        super(ResUP, self).__init__()
        self.in_channel_0 = channels[0] 
        self.in_channel_1 = channels[1] 
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        self.res_mode = res
        self.att = CBAM(in_ch=channels[1], rate=4)
        self.fuse_conv2 = nn.Sequential(
            Conv(inp_dim=channels[1]*2, out_dim=channels[1], bn=True),
            Conv(inp_dim=channels[1], out_dim=channels[1], bn=False)
        )

    def forward(self, x1, x2):
        h = x2.shape[2]
        w = x2.shape[3]
        x1_up = F.interpolate(self.fuse_conv1(x1), size=[h, w], mode='bilinear', align_corners=False)
        hi_3 = self.fuse_conv2(torch.cat([x1_up, x2], dim=1))
        x_out = self.att(hi_3)
        if self.res_mode == 1:
            x_out = x_out + x2
        return x_out


class MTU(nn.Module):
    def __init__(self,
                 in_channels=3, out_channels=2, img_size=384,
                 patch_size=16, hidden_size=768, num_heads=12, mlp_dim=3072,
                 feature_size=16):
        super(MTU, self).__init__()
        down_times = int(img_size / patch_size)
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.fri_down = nn.MaxPool2d(kernel_size=down_times, stride=down_times)
        self.patch_emb = PatchEmbedding(in_channels, patch_size=patch_size, emb_size=hidden_size, img_size=patch_size)
        self.encoder = nn.ModuleList(
            [ViTBlock(emb_size=hidden_size, n_heads=num_heads, mlp_dim=mlp_dim) for _ in range(12)]
        )
        self.HFS_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size, out_channels=feature_size*2, kernel_size=1, bias=True),
            nn.BatchNorm2d(feature_size*2),
            nn.ReLU()
        )
        self.HFS_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size, out_channels=feature_size, kernel_size=1, bias=True),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )
        self.HFS_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size, out_channels=feature_size, kernel_size=1, bias=True),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )
        self.HFS_fc = HFS(channels=[feature_size, feature_size, feature_size*2], res=1)
        self.multi_res_1 = SpatialMap(in_ch=feature_size * 2, out_ch=feature_size * 32, spa_size=16)
        self.multi_res_2 = SpatialMap(in_ch=hidden_size, out_ch=feature_size * 16, spa_size=32)
        self.multi_res_3 = SpatialMap(in_ch=feature_size, out_ch=feature_size * 8, spa_size=64)
        self.multi_res_4 = SpatialMap(in_ch=feature_size, out_ch=feature_size * 4, spa_size=128)
        self.encoder1 = nn.Sequential(
            Conv(inp_dim=in_channels, out_dim=feature_size, bn=True),
            Conv(inp_dim=feature_size, out_dim=feature_size, bn=True)
        )
        self.att = nn.Sequential(
            CBAM(in_ch=feature_size, rate=4),
            nn.BatchNorm2d(feature_size)
        )
        self.decoder = ResUP(channels=[feature_size * 2, feature_size], res=1)
        self.out = nn.Sequential(
            Conv(inp_dim=feature_size, out_dim=feature_size, bn=True),
            Conv(inp_dim=feature_size, out_dim=1, bn=False)
        )
        self.one_hot = nn.Sigmoid()
        self.conv_based_block = CBB(in_ch=in_channels, out_ch=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x_in):
        x_pd = self.fri_down(x_in) 
        x = self.patch_emb(x_pd) 
        enc_features = []
        for layer in self.encoder:
            x = layer(x)
            enc_features.append(x)

        HFS_coding = []
        x12 = self.HFS_conv0(self.proj_feat(x, hidden_size=self.hidden_size, feat_size=[1, 1])) 
        x6 = self.HFS_conv1(self.proj_feat(enc_features[5], hidden_size=self.hidden_size, feat_size=[1, 1]))  
        x3 = self.HFS_conv2(self.proj_feat(enc_features[2], hidden_size=self.hidden_size, feat_size=[1, 1]))  

        HFS_coding.append(x3)
        HFS_coding.append(x6)
        dec4 = F.interpolate(x12, size=[8, 8], mode='bilinear', align_corners=False)  
        HFS_coding.append(dec4)
        dec4 = self.HFS_fc(HFS_coding) 

        enc1 = self.att(self.encoder1(x_pd)) 
        out = self.out(self.decoder(dec4, enc1))
        out = F.interpolate(out, size=[self.img_size, self.img_size], mode='bilinear', align_corners=False)

        fri_out = self.one_hot(out)
        trans_out = x_in * fri_out + x_in

        en16 = self.multi_res_1(dec4)
        en32 = self.multi_res_2(self.proj_feat(enc_features[8], hidden_size=self.hidden_size,
                                           feat_size=[1, 1]))  
        en64 = self.multi_res_3(x6)  
        en128 = self.multi_res_4(x3) 

        final_out = self.conv_based_block(trans_out, [en128, en64, en32, en16]) 

        return final_out
