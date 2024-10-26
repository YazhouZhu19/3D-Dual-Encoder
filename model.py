import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class encoder_network(nn.Module):

    def __init__(self):
        super(encoder_network, self).__init__()
        self.conv0_1 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, stride = 1, padding = 1, bias=False)
        self.norm0_1 = nn.GroupNorm(num_groups=1, num_channels = 8)
        self.actv0_1 = nn.LeakyReLU(inplace=True)

        self.ds_1 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv1_1 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1_1 = nn.GroupNorm(num_groups=1, num_channels = 16)
        self.actv1_1 = nn.LeakyReLU(inplace=True)

        self.ds_2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv2_1 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2_1 = nn.GroupNorm(num_groups=1, num_channels=32)
        self.actv2_1 = nn.LeakyReLU(inplace=True)

        self.ds_3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv3_1 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3_1 = nn.GroupNorm(num_groups=1, num_channels=64)
        self.actv3_1 = nn.LeakyReLU(inplace=True)

        self.ds_4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv4_1 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm4_1 = nn.GroupNorm(num_groups=1, num_channels=128)
        self.actv4_1 = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        out_0 = self.conv0_1(x)
        out_0 = self.norm0_1(out_0)
        out_0 = self.actv0_1(out_0)

        out_1_ = self.ds_1(out_0)

        out_1 = self.conv1_1(out_1_)
        out_1 = self.norm1_1(out_1)
        out_1 = self.actv1_1(out_1)

        out_2_ = self.ds_2(out_1)

        out_2 = self.conv2_1(out_2_)
        out_2 = self.norm2_1(out_2)
        out_2 = self.actv2_1(out_2)

        out_3_ = self.ds_3(out_2)

        out_3 = self.conv3_1(out_3_)
        out_3 = self.norm3_1(out_3)
        out_3 = self.actv3_1(out_3)

        out_4_ = self.ds_4(out_3)

        out_4 = self.conv4_1(out_4_)
        out_4 = self.norm4_1(out_4)
        out_4 = self.actv4_1(out_4)

        out = out_4

        return out_0, out_1_, out_2_, out_3_, out_4_, out


class attention(nn.Module):

    def __init__(self):
        super(attention, self).__init__()

        self.resize_1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)

        self.channelConv = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.weightConv = nn.Conv3d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        self.resize_2 = nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x, y):

        scale_1 = self.resize_1(x)  # (1, 64, 5, 5, 5)
        scale_2 = y  # (1, 64, 5, 5, 5)

        scale_sum = torch.cat((scale_1, scale_2), 1)  # (1, 128, 5, 5, 5)
        scale_sum = self.channelConv(scale_sum)  # (1, 64, 5, 5, 5)

        scale_sum = self.weightConv(scale_sum)  # (1, 2, 5, 5, 5)

        weight_1 = scale_sum[:, 0, :, :, :].view(2, 1, 5, 5, 5)  # (1, 1, 5, 5, 5)
        weight_2 = scale_sum[:, 1, :, :, :].view(2, 1, 5, 5, 5)  # (1, 1, 5, 5, 5)

        weight_1 = self.resize_2(weight_1)

        weight_1 = F.softmax(weight_1, dim=1)  # (1, 1, 10, 10, 10)
        weight_2 = F.softmax(weight_2, dim=1)  # (1, 1, 5, 5, 5)

        return weight_1, weight_2


class seg_net(nn.Module):

    def __init__(self):
        super(seg_net, self).__init__()
        self.resize_conv_1 = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1, bias=False)

        self.encoder_scale_1 = encoder_network()  # (1, 64, 20, 20, 20)
        self.encoder_scale_2 = encoder_network()  # (1, 64, 10, 10, 10)

        # **********************************************************************************************
        self.ds_1 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False)
        self.merge_1 = nn.Conv3d(in_channels=24, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

        self.ds_2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        self.merge_2 = nn.Conv3d(in_channels=48, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)

        self.ds_3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.merge_3 = nn.Conv3d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # ***********************************************************************************************

        self.attentionModel = attention()

        self.resize_conv_2 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=False)
        self.channelConv = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        # decoder network
        self.dn_conv0 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)  # (1, 256, 10, 10, 10)

        self.us_1 = nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False)  # (1, 256, 20, 20, 20)
        self.dn_conv1_1 = nn.Conv3d(in_channels=288, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)  # (1, 128, 20, 20, 20)
        self.dn_norm1_1 = nn.GroupNorm(num_groups=1, num_channels=128)
        self.dn_actv1_1 = nn.LeakyReLU(inplace=True)
        self.dn_conv1_2 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dn_norm1_2 = nn.GroupNorm(num_groups=1, num_channels=128)
        self.dn_actv1_2 = nn.LeakyReLU(inplace=True)

        self.us_2 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=False)  # (1, 128, 40, 40, 40)
        self.dn_conv2_1 = nn.Conv3d(in_channels=144, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)  # (1, 64, 40, 40, 40)
        self.dn_norm2_1 = nn.GroupNorm(num_groups=1, num_channels=64)
        self.dn_actv2_1 = nn.LeakyReLU(inplace=True)
        self.dn_conv2_2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dn_norm2_2 = nn.GroupNorm(num_groups=1, num_channels=64)
        self.dn_actv2_2 = nn.LeakyReLU(inplace=True)

        self.us_3 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=False)  # (1, 64, 80, 80, 80)
        self.dn_conv3_1 = nn.Conv3d(in_channels=72, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)  # (1, 64, 80, 80, 80)
        self.dn_norm3_1 = nn.GroupNorm(num_groups=1, num_channels=32)
        self.dn_actv3_1 = nn.LeakyReLU(inplace=True)
        self.dn_conv3_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dn_norm3_2 = nn.GroupNorm(num_groups=1, num_channels=32)
        self.dn_actv3_2 = nn.LeakyReLU(inplace=True)

        self.us_4 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, bias=False)  # (1, 32, 160, 160, 160)
        self.dn_conv4_1 = nn.Conv3d(in_channels=40, out_channels=16, kernel_size=3,  stride=1, padding=1, bias=False)  # (1, 16, 160, 160, 160)
        self.dn_norm4_1 = nn.GroupNorm(num_groups=1, num_channels=16)
        self.dn_actv4_1 = nn.LeakyReLU(inplace=True)
        self.dn_conv4_2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.dn_norm4_2 = nn.GroupNorm(num_groups=1, num_channels=16)
        self.dn_actv4_2 = nn.LeakyReLU(inplace=True)

        self.outputConv = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        image_1 = x  # (1, 4, 160, 160, 160)
        image_2 = self.resize_conv_1(x)  # (1, 4, 80, 80, 80)

        feature_map_a_0, feature_map_a_1_, feature_map_a_2_, feature_map_a_3_, feature_map_a_4_, feature_map_a = self.encoder_scale_1(image_1)
        # 1:(1, 8, 80, 80, 80)  2:(1, 16, 40, 40, 40)  3:(1, 32, 20, 20, 20) 4:(1, 64, 10, 10, 10)   a:(1, 128, 10, 10, 10)

        feature_map_a_1 = self.ds_1(feature_map_a_1_)
        feature_map_a_2 = torch.cat((feature_map_a_1, feature_map_a_2_), 1)
        feature_map_a_2 = self.merge_1(feature_map_a_2)  # (1, 16, 40, 40, 40)

        feature_map_a_2 = self.ds_2(feature_map_a_2)
        feature_map_a_3 = torch.cat((feature_map_a_2, feature_map_a_3_), 1)
        feature_map_a_3 = self.merge_2(feature_map_a_3)

        feature_map_a_3 = self.ds_3(feature_map_a_3)
        feature_map_a_4 = torch.cat((feature_map_a_3, feature_map_a_4_), 1)
        feature_map_a_4 = self.merge_3(feature_map_a_4)  # (1, 64, 10, 10, 10)  ***

        feature_map_b_0, feature_map_b_1_, feature_map_b_2_, feature_map_b_3_, feature_map_b_4_, feature_map_b = self.encoder_scale_2(image_2)
        # 1:(1, 8, 40, 40, 40)  2:(1, 16, 20, 20, 20)  3:(1, 32, 10, 10, 10) 4:(1, 64, 5, 5, 5)   b:(1, 128, 5, 5, 5)

        feature_map_b_1 = self.ds_1(feature_map_b_1_)
        feature_map_b_2 = torch.cat((feature_map_b_1, feature_map_b_2_), 1)
        feature_map_b_2 = self.merge_1(feature_map_b_2)  # (1, 16, 20, 20, 20)

        feature_map_b_2 = self.ds_2(feature_map_b_2)
        feature_map_b_3 = torch.cat((feature_map_b_2, feature_map_b_3_), 1)
        feature_map_b_3 = self.merge_2(feature_map_b_3)

        feature_map_b_3 = self.ds_3(feature_map_b_3)
        feature_map_b_4 = torch.cat((feature_map_b_3, feature_map_b_4_), 1)
        feature_map_b_4 = self.merge_3(feature_map_b_4)  # (1, 64, 5, 5, 5)  ***

        weight1, weight2 = self.attentionModel(feature_map_a_4, feature_map_b_4)  # (1, 1, 10, 10, 10) (1, 1, 5, 5, 5)


        feature_map_1 = feature_map_a * weight1  # (1, 128, 10, 10, 10)
        feature_map_2 = feature_map_b * weight2  # (1, 128, 5, 5, 5)

        feature_map_2 = self.resize_conv_2(feature_map_2)  # (1, 128, 10, 10, 10)

        feature_map = torch.cat((feature_map_1, feature_map_2), 1)  # (1, 256, 10, 10, 10)
        feature_map = self.channelConv(feature_map)  # (1, 128, 10, 10, 10)


        # decoder network
        dn_0 = self.dn_conv0(feature_map)  # (1, 256, 10, 10, 10)

        dn_1 = self.us_1(dn_0)  # (1, 256, 20, 20, 20)
        dn_1 = torch.cat((dn_1, feature_map_a_3_), 1)  # (1, 288, 20, 20, 20)
        dn_1 = self.dn_conv1_1(dn_1)  # (1, 128, 20, 20, 20)
        dn_1 = self.dn_norm1_1(dn_1)
        dn_1 = self.dn_actv1_1(dn_1)
        dn_1 = self.dn_conv1_2(dn_1)
        dn_1 = self.dn_norm1_2(dn_1)
        dn_1 = self.dn_actv1_2(dn_1)

        dn_2 = self.us_2(dn_1)  # (1, 128, 40, 40, 40)
        dn_2 = torch.cat((dn_2, feature_map_a_2_), 1)  # (1, 144, 40, 40, 40)
        dn_2 = self.dn_conv2_1(dn_2)  # (1, 64, 40, 40, 40)
        dn_2 = self.dn_norm2_1(dn_2)
        dn_2 = self.dn_actv2_1(dn_2)
        dn_2 = self.dn_conv2_2(dn_2)
        dn_2 = self.dn_norm2_2(dn_2)
        dn_2 = self.dn_actv2_2(dn_2)

        dn_3 = self.us_3(dn_2)  # (1, 64, 80, 80, 80)
        dn_3 = torch.cat((dn_3, feature_map_a_1_), 1)
        dn_3 = self.dn_conv3_1(dn_3)
        dn_3 = self.dn_norm3_1(dn_3)
        dn_3 = self.dn_actv3_1(dn_3)
        dn_3 = self.dn_conv3_2(dn_3)
        dn_3 = self.dn_norm3_2(dn_3)
        dn_3 = self.dn_actv3_2(dn_3)

        dn_4 = self.us_4(dn_3)
        dn_4 = torch.cat((dn_4, feature_map_a_0), 1)
        dn_4 = self.dn_conv4_1(dn_4)
        dn_4 = self.dn_norm4_1(dn_4)
        dn_4 = self.dn_actv4_1(dn_4)
        dn_4 = self.dn_conv4_2(dn_4)
        dn_4 = self.dn_norm4_2(dn_4)
        dn_4 = self.dn_actv4_2(dn_4)

        out = self.outputConv(dn_4)

        return out



