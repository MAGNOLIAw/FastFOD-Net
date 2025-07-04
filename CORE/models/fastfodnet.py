import torch
import torch.nn.functional as F
import torch.nn as nn

class GateConv(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 activation=torch.nn.LeakyReLU(0.2, inplace=True), use_gate=False, norm_layer=nn.BatchNorm3d):
        super(GateConv, self).__init__()

        self.activation = activation
        self.use_gate = use_gate

        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 
                                      padding, dilation, groups, bias)
        self.mask_conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 
                                           padding, dilation, groups, bias)
        self.batch_norm3d = norm_layer(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv3d(input)
        mask = self.mask_conv3d(input)

        x = self.batch_norm3d(x)
        if self.activation is not None:
            x = self.activation(x)
        # gated features
        if self.use_gate:
            x = x * self.gated(mask)        
        return x


class GateDeConv(torch.nn.Module):
    """
    3D Gated Deconvolution (Transposed Convolution) with optional normalization and activation.
    """    
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=2, 
                 padding=1, dilation=1, groups=1, bias=True, 
                 activation=torch.nn.LeakyReLU(0.2, inplace=True),
                 norm_layer=nn.BatchNorm3d):
        super(GateDeConv, self).__init__()

        # self.conv3d = GateConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation, use_gate=False)
        # self.scale_factor = scale_factor

        self.conv3d2 = nn.Sequential(
                        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=1),
                        norm_layer(out_channels),
                        activation)


    def forward(self, input):
        # x = F.interpolate(input, scale_factor=self.scale_factor)
        # x = self.conv3d(x)

        x = self.conv3d2(input)
        return x


class FastFODNet(nn.Module):
    """
    FastFOD-Net: A 3D UNet-based network with gated convolutions.
    """
    def __init__(self, in_dim=45, out_dim=45, cnum=32, norm_layer=nn.BatchNorm2d, activation = nn.ReLU(inplace=True)):
        super(FastFODNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        # activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.enc1_1 = GateConv(in_dim, cnum, 3, 1, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)
        self.enc1_2 = GateConv(cnum, cnum, 3, 2, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)
        # downsample to 128
        self.enc2_1 = GateConv(cnum, 2 * cnum, 3, 1, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)
        self.enc2_2 = GateConv(2 * cnum, 2 * cnum, 3, 2, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)
        # downsample to 64
        self.enc3_1 = GateConv(2 * cnum, 4 * cnum, 3, 1, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)
        self.enc3_2 = GateConv(4 * cnum, 4 * cnum, 3, 2, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)
        # downsample to 32
        self.enc4_1 = GateConv(4 * cnum, 8 * cnum, 3, 1, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)
        self.enc4_2 = GateConv(8 * cnum, 8 * cnum, 3, 2, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)

        # Bridge
        self.bridge = GateConv(8 * cnum, 16 * cnum, 3, 1, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)

        # Up sampling
        self.dec1_1 = GateDeConv(2, 16 * cnum, 8 * cnum, 3, 2, padding=1, activation=activation)
        self.dec1_2 = GateConv(16 * cnum, 8 * cnum, 3, 1, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)
        self.dec2_1 = GateDeConv(2, 8 * cnum, 4 * cnum, 3, 2, padding=1, activation=activation)
        self.dec2_2 = GateConv(8 * cnum, 4 * cnum, 3, 1, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)
        self.dec3_1 = GateDeConv(2, 4 * cnum, 2 * cnum, 3, 2, padding=1, activation=activation)
        self.dec3_2 = GateConv(4 * cnum, 2 * cnum, 3, 1, padding=1,activation=activation, use_gate=False, norm_layer=norm_layer)
        self.dec4_1 = GateDeConv(2, 2 * cnum, cnum, 3, 2, padding=1, activation=activation)
        self.dec4_2 = GateConv(2 * cnum, cnum, 3, 1, padding=1, activation=activation, use_gate=False, norm_layer=norm_layer)

        # Output
        self.out = GateConv(cnum, out_dim, 3, 1, padding=1, activation=None, use_gate=False)

    def forward(self, x):
        # x: b c w h d

        # Down sampling
        down_1 = self.enc1_1(x)
        pool_1 = self.enc1_2(down_1)

        down_2 = self.enc2_1(pool_1)
        pool_2 = self.enc2_2(down_2)

        down_3 = self.enc3_1(pool_2)
        pool_3 = self.enc3_2(down_3)

        down_4 = self.enc4_1(pool_3)
        pool_4 = self.enc4_2(down_4)

        # Bridge
        bridge = self.bridge(pool_4)

        # Up sampling
        trans_1 = self.dec1_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.dec1_2(concat_1)

        trans_2 = self.dec2_1(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.dec2_2(concat_2)

        trans_3 = self.dec3_1(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.dec3_2(concat_3)

        trans_4 = self.dec4_1(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.dec4_2(concat_4)

        # Output
        out = self.out(up_4)

        return out
