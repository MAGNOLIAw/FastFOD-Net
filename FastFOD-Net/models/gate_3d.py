import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    pad = int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)
    return pad

def pad_to(x, stride):
    h, w, d = x.shape[-3:]
    # print(h,w,d)

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    if d % stride > 0:
        new_d = d + stride - d % stride
    else:
        new_d = d
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    ld, ud = int((new_d - d) / 2), int(new_d - d) - int((new_d - d) / 2)
    pads = (ld, ud, lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[4]+pad[5] > 0:
        x = x[:,:,pad[4]:-pad[5],:,:]
    if pad[2]+pad[3] > 0:
        x = x[:,:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,:,pad[0]:-pad[1]]
    return x

class LesionGateConv(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), use_gate=True):
        super(LesionGateConv, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm3d = torch.nn.BatchNorm3d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.use_gate = use_gate

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv3d(input)
        mask = self.mask_conv3d(input)

        # gated features
        if self.batch_norm:
            x = self.batch_norm3d(x)

        if self.activation is not None:
            if self.use_gate:
                x = self.activation(x) * self.gated(mask)
            else:
                x = self.activation(x)
        else:
            if self.use_gate:
                x = x * self.gated(mask)
        return x


class LesionGateDeConv(torch.nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(LesionGateDeConv, self).__init__()
        self.conv3d = LesionGateConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation, use_gate=True)
        self.scale_factor = scale_factor

        # self.conv3d2 = nn.Sequential(
        #                 nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1),
        #                 nn.BatchNorm3d(out_channels),
        #                 activation)

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        x = self.conv3d(x)
        # x = self.conv3d2(input)
        return x

def max_pooling_3d(ks=(2, 2, 2), stride=(2, 2, 2)):
    return nn.MaxPool3d(kernel_size=ks, stride=stride, padding=0)

class GateUNet3D(nn.Module):
    def __init__(self, in_dim=1, out_dim=2, batch_norm=True, cnum=32):
        super(GateUNet3D, self).__init__()
        print('GateUNet3D')
        self.in_dim = in_dim
        self.out_dim = out_dim
        # activation = nn.ReLU(inplace=True)
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        div = 4
        self.enc1_1 = LesionGateConv(in_dim, cnum, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        self.enc1_2 = LesionGateConv(cnum, cnum, 3, 2, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        # self.pool_1 = max_pooling_3d()
        # downsample to 128
        self.enc2_1 = LesionGateConv(cnum, 2 * cnum, 3, 1, padding=get_pad(128 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        self.enc2_2 = LesionGateConv(2 * cnum, 2 * cnum, 3, 2, padding=get_pad(128 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        # self.pool_2 = max_pooling_3d()
        # downsample to 64
        self.enc3_1 = LesionGateConv(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(64 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        self.enc3_2 = LesionGateConv(4 * cnum, 4 * cnum, 3, 2, padding=get_pad(64 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        # self.pool_3 = max_pooling_3d()
        # downsample to 32
        self.enc4_1 = LesionGateConv(4 * cnum, 8 * cnum, 3, 1, padding=get_pad(32 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        self.enc4_2 = LesionGateConv(8 * cnum, 8 * cnum, 3, 2, padding=get_pad(32 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        # self.pool_4 = max_pooling_3d()

        # Bridge
        self.bridge = LesionGateConv(8 * cnum, 16 * cnum, 3, 1, padding=get_pad(16 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)

        # Up sampling
        self.dec1_1 = LesionGateDeConv(2, 16 * cnum, 8 * cnum, 3, 1, padding=get_pad(32 // div, 3, 1), batch_norm=batch_norm, activation=activation)
        self.dec1_2 = LesionGateConv(16 * cnum, 8 * cnum, 3, 1, padding=get_pad(32 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        self.dec2_1 = LesionGateDeConv(2, 8 * cnum, 4 * cnum, 3, 1, padding=get_pad(64 // div, 3, 1), batch_norm=batch_norm, activation=activation)
        self.dec2_2 = LesionGateConv(8 * cnum, 4 * cnum, 3, 1, padding=get_pad(64 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        self.dec3_1 = LesionGateDeConv(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128 // div, 3, 1), batch_norm=batch_norm, activation=activation)
        self.dec3_2 = LesionGateConv(4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)
        self.dec4_1 = LesionGateDeConv(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm, activation=activation)
        self.dec4_2 = LesionGateConv(2 * cnum, cnum, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm, activation=activation, use_gate=True)

        # Output
        self.out = LesionGateConv(cnum, out_dim, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm, activation=None, use_gate=True)
        # self.out1 = LesionGateConv(cnum, 1, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm, activation=None, use_gate=True)
        # self.out2 = LesionGateConv(cnum, 5, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm, activation=None, use_gate=True)
        # self.out3 = LesionGateConv(cnum, 9, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm, activation=None, use_gate=True)
        # self.out4 = LesionGateConv(cnum, 13, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm, activation=None, use_gate=True)
        # self.out5 = LesionGateConv(cnum, 17, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm, activation=None, use_gate=True)


    def forward(self, x, encoder_only=False, save_feat=False, lgc_layers=['enc4_1', 'enc3_1', 'enc2_1']):
        feat = []

        # print(x.shape)
        x, pads = pad_to(x, 16)  # Padded data, feed this to your network

        # x: b c w h d
        # Down sampling
        down_1 = self.enc1_1(x)
        pool_1 = self.enc1_2(down_1)
        # pool_1 = self.pool_1(down_1)  # -> [1, 4, 64, 64, 64]

        down_2 = self.enc2_1(pool_1)
        pool_2 = self.enc2_2(down_2)
        # pool_2 = self.pool_2(down_2)  # -> [1, 8, 32, 32, 32]
        x = pool_2
        if 'enc2_1' in lgc_layers:
            feat.append(x)

        down_3 = self.enc3_1(pool_2)
        pool_3 = self.enc3_2(down_3)
        # pool_3 = self.pool_3(down_3)  # -> [1, 16, 16, 16, 16]
        x = pool_3
        if 'enc3_1' in lgc_layers:
            feat.append(x)

        down_4 = self.enc4_1(pool_3)
        pool_4 = self.enc4_2(down_4)
        # pool_4 = self.pool_4(down_4)  # -> [1, 32, 8, 8, 8]
        x = pool_4
        if 'enc4_1' in lgc_layers:
            feat.append(x)
        # print('pool shape', pool_1.shape, pool_2.shape, pool_3.shape, pool_4.shape)

        if encoder_only:
            return feat

        # Bridge
        bridge = self.bridge(pool_4)

        # Up sampling
        trans_1 = self.dec1_1(bridge)  # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_4], dim=1)  # -> [1, 192, 8, 8, 8]
        up_1 = self.dec1_2(concat_1)  # -> [1, 64, 8, 8, 8]

        trans_2 = self.dec2_1(up_1)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_3], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.dec2_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.dec3_1(up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_2], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.dec3_2(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.dec4_1(up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_1], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.dec4_2(concat_4)  # -> [1, 8, 64, 64, 64]
        # print('up shape', up_1.shape, up_2.shape, up_3.shape, up_4.shape)

        # Output
        out = self.out(up_4)  # -> [1, 3, 128, 128, 128]
        out = unpad(out, pads)
        # # predict by 5 orders
        # o1 = self.out1(up_4)
        # o2 = self.out2(up_4)
        # o3 = self.out3(up_4)
        # o4 = self.out4(up_4)
        # o5 = self.out5(up_4)
        # out1 = torch.cat([o1, o2, o3, o4, o5], dim=1)
        # out = out1

        if save_feat:
            # print('save feat', len(feat))
            return out, feat
        else:
            return out
