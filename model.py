from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb



class InitialBlock(nn.Module):
    '''

    The initial block for Enet has 2 branches: The convolution branch and
    maxpool branch.
    The conv branch has 13 layers, while the maxpool branch gives 3 layers
    corresponding to the RBG channels.
    Both output layers are then concatenated to give an output of 16 layers.
    INPUTS:
    - input(Tensor): A 4D tensor of shape [batch_size, channel, height, width]
    '''


    def __init__(self):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, 13, (3, 3), stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(13, 1e-3)
        self.prelu = nn.PReLU(13)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, input):
        output = torch.cat([
            self.prelu(self.batch_norm(self.conv(input))), self.pool(input)
        ], 1)
        return output


class BottleNeck(nn.Module):
    '''

    The bottle module has three different kinds of variants:
    1. A regular convolution which you can decide whether or not to downsample.
    2. A dilated convolution which requires you to have a dilation factor.
    3. An asymetric convolution that has a decomposed filter size of 5x1 and
    1x5 separately.
    INPUTS:
    - inputs(Tensor): a 4D Tensor of the previous convolutional block of shape
    [batch_size, channel, height, widht].
    - output_channels(int): an integer indicating the output depth of the
    output convolutional block.
    - regularlizer_prob(float): the float p that represents the prob of
    dropping a layer for spatial dropout regularlization.
    - downsampling(bool): if True, a max-pool2D layer is added to downsample
    the spatial sizes.
    - upsampling(bool): if True, the upsampling bottleneck is activated but
    requires pooling indices to upsample.
    - dilated(bool): if True, then dilated convolution is done, but requires
    a dilation rate to be given.
    - dilation_rate(int): the dilation factor for performing atrous
    convolution/dilated convolution
    - asymmetric(bool): if True, then asymmetric convolution is done, and
    the only filter size used here is 5.
    - use_relu(bool): if True, then all the prelus become relus according to
    Enet author.
    '''


    def __init__(self,
                 input_channels=None,
                 output_channels=None,
                 regularlizer_prob=0.1,
                 downsampling=False,
                 upsampling=False,
                 dilated=False,
                 dilation_rate=None,
                 asymmetric=False,
                 use_relu=False):
        super(BottleNeck, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.use_relu = use_relu

        internal = output_channels // 4
        input_stride = 2 if downsampling else 1
        # First projection with 1x1 kernel (2x2 for downsampling)
        conv1x1_1 = nn.Conv2d(input_channels, internal,
                              input_stride, input_stride, bias=False)
        batch_norm1 = nn.BatchNorm2d(internal, 1e-3)
        prelu1 = self._prelu(internal, use_relu)
        self.block1x1_1 = nn.Sequential(conv1x1_1, batch_norm1, prelu1)

        conv = None
        if downsampling:
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
            conv = nn.Conv2d(internal, internal, 3, stride=1, padding=1)
        elif upsampling:
            # padding is replaced with spatial convolution without bias.
            spatial_conv = nn.Conv2d(input_channels, output_channels, 1,
                                     bias=False)
            batch_norm = nn.BatchNorm2d(output_channels, 1e-3)
            self.conv_before_unpool = nn.Sequential(spatial_conv, batch_norm)
            self.unpool = nn.MaxUnpool2d(2)
            conv = nn.ConvTranspose2d(internal, internal, 3,
                                      stride=2, padding=1, output_padding=1)
        elif dilated:
            conv = nn.Conv2d(internal, internal, 3, padding=dilation_rate,
                             dilation=dilation_rate)
        elif asymmetric:
            conv1 = nn.Conv2d(internal, internal, [5, 1], padding=(2, 0),
                              bias=False)
            conv2 = nn.Conv2d(internal, internal, [1, 5], padding=(0, 2))
            conv = nn.Sequential(conv1, conv2)
        else:
            conv = nn.Conv2d(internal, internal, 3, padding=1)

        batch_norm = nn.BatchNorm2d(internal, 1e-3)
        prelu = self._prelu(internal, use_relu)
        self.middle_block = nn.Sequential(conv, batch_norm, prelu)

        # Final projection with 1x1 kernel
        conv1x1_2 = nn.Conv2d(internal, output_channels, 1, bias=False)
        batch_norm2 = nn.BatchNorm2d(output_channels, 1e-3)
        prelu2 = self._prelu(output_channels, use_relu)
        self.block1x1_2 = nn.Sequential(conv1x1_2, batch_norm2, prelu2)

        # regularlize
        self.dropout = nn.Dropout2d(regularlizer_prob)

    def _prelu(self, channels, use_relu):
        return (nn.PReLU(channels) if use_relu is False else nn.ReLU())

    def forward(self, input, pooling_indices=None):
        main = None
        input_shape = input.size()
        if self.downsampling:
            main, indices = self.pool(input)
            if (self.output_channels != self.input_channels):
                pad = Variable(torch.Tensor(input_shape[0],
                               self.output_channels - self.input_channels,
                               input_shape[2] // 2,
                               input_shape[3] // 2).zero_(), requires_grad=False)
                if (torch.cuda.is_available):
                    pad = pad.cuda(0)
                main = torch.cat((main, pad), 1)
        elif self.upsampling:
            main = self.unpool(self.conv_before_unpool(input), pooling_indices)
        else:
            main = input

        other_net = nn.Sequential(self.block1x1_1, self.middle_block,
                                  self.block1x1_2)
        other = other_net(input)
        output = F.relu(main + other)
        if (self.downsampling):
            return output, indices
        return output

ENCODER_LAYER_NAMES = ['initial', 'bottleneck_1_0', 'bottleneck_1_1',
                       'bottleneck_1_2', 'bottleneck_1_3', 'bottleneck_1_4',
                       'bottleneck_2_0', 'bottleneck_2_1', 'bottleneck_2_2',
                       'bottleneck_2_3', 'bottleneck_2_4', 'bottleneck_2_5',
                       'bottleneck_2_6', 'bottleneck_2_7', 'bottleneck_2_8',
                       'bottleneck_3_1', 'bottleneck_3_2', 'bottleneck_3_3',
                       'bottleneck_3_4', 'bottleneck_3_5', 'bottleneck_3_6',
                       'bottleneck_3_7', 'bottleneck_3_8', 'classifier']
DECODER_LAYER_NAMES = ['bottleneck_4_0', 'bottleneck_4_1', 'bottleneck_4_2'
                       'bottleneck_5_0', 'bottleneck_5_1', 'fullconv']


class Encoder(nn.Module):
    def __init__(self, num_classes, only_encode=True):
        super(Encoder, self).__init__()
        self.state = only_encode
        layers = []
        layers.append(InitialBlock())
        layers.append(BottleNeck(16, 64, regularlizer_prob=0.01,
                                 downsampling=True))
        for i in range(4):
            layers.append(BottleNeck(64, 64, regularlizer_prob=0.01))
        
        # Section 2 and 3
        layers.append(BottleNeck(64, 128, downsampling=True))
        for i in range(2):
            layers.append(BottleNeck(128, 128))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=2))
            layers.append(BottleNeck(128, 128, asymmetric=True))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=4))
            layers.append(BottleNeck(128, 128))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=8))
            layers.append(BottleNeck(128, 128, asymmetric=True))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=16))
        # only training encoder
        if only_encode:
            layers.append(nn.Conv2d(128, num_classes, 1))

        for layer, layer_name in zip(layers, ENCODER_LAYER_NAMES):
            super(Encoder, self).__setattr__(layer_name, layer)
        self.layers = layers

    
    def forward(self, input):
        pooling_stack = []
        output = input
        for layer in self.layers:
            if hasattr(layer, 'downsampling') and layer.downsampling:
                output, pooling_indices = layer(output)
                pooling_stack.append(pooling_indices)
            else:
                output = layer(output)

        if self.state:
            output = F.upsample(output, cfg.TRAIN.IMG_SIZE, None, 'bilinear')

        return output, pooling_stack


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        layers = []
        # Section 4
        layers.append(BottleNeck(128, 64, upsampling=True, use_relu=True))
        layers.append(BottleNeck(64, 64, use_relu=True))
        layers.append(BottleNeck(64, 64, use_relu=True))

        # Section 5
        layers.append(BottleNeck(64, 16, upsampling=True, use_relu=True))
        layers.append(BottleNeck(16, 16, use_relu=True))
        layers.append(nn.ConvTranspose2d(16, num_classes, 2, stride=2))

        self.layers = nn.ModuleList([layer for layer in layers])
    
    def forward(self, input, pooling_stack):
        output = input
        for layer in self.layers:
            if hasattr(layer, 'upsampling') and layer.upsampling:
                pooling_indices = pooling_stack.pop()
                output = layer(output, pooling_indices)
            else:
                output = layer(output)
        return output





import torch
from torch import nn
from build_contextpath import build_contextpath
import warnings
warnings.filterwarnings(action='ignore')

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from context path) + 1024(from spatial path) + 2048(from spatial path)
        # resnet18  1024 = 256(from context path) + 256(from spatial path) + 512(from spatial path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)

        # build attention refinement module  for resnet 101
        if context_path == 'resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

        elif context_path == 'resnet18':
            # build attention refinement module  for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
        else:
            print('Error: unspport context_path network \n')

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.conv)

    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)

        if self.training == True:
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear')
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear')

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)

        if self.training == True:
            return result, cx1_sup, cx2_sup

        return result

class ENet(nn.Module):
    def __init__(self, only_encode=False):
        super(ENet, self).__init__()
        self.state = only_encode
        self.encoder = Encoder(cfg.DATA.NUM_CLASSES,only_encode=only_encode)
        self.decoder = Decoder(cfg.DATA.NUM_CLASSES)

    def forward(self, input):
        output, pooling_stack = self.encoder(input)
        if not self.state:
            output = self.decoder(output, pooling_stack)
        return output
        






"""Image Cascade Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from segbase import SegBaseModel
from torchsummary import summary

__all__ = ['ICNet', 'get_icnet', 'get_icnet_resnet50_citys',
           'get_icnet_resnet101_citys', 'get_icnet_resnet152_citys']

class ICNet(SegBaseModel):
    """Image Cascade Network"""
    
    def __init__(self, nclass = 5, backbone='resnet50', pretrained_base=True):
        super(ICNet, self).__init__(nclass,backbone, pretrained_base=pretrained_base)
        self.conv_sub1 = nn.Sequential(
            _ConvBNReLU(3, 32, 3, 2),
            _ConvBNReLU(32, 32, 3, 2),
            _ConvBNReLU(32, 64, 3, 2)
        )
        
        self.ppm = PyramidPoolingModule()

        self.head = _ICHead(nclass)

        self.__setattr__('exclusive', ['conv_sub1', 'head'])
        
    def forward(self, x):
        # sub 1
        x_sub1 = self.conv_sub1(x)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        _, x_sub2, _, _ = self.base_forward(x_sub2)
        
        # sub 4
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        _, _, _, x_sub4 = self.base_forward(x_sub4)
        # add PyramidPoolingModule
        x_sub4 = self.ppm(x_sub4)
        
        outputs = self.head(x_sub1, x_sub2, x_sub4)
        
        return tuple(outputs)

class PyramidPoolingModule(nn.Module):
	def __init__(self, pyramids=[1,2,3,6]):
		super(PyramidPoolingModule, self).__init__()
		self.pyramids = pyramids

	def forward(self, input):
		feat = input
		height, width = input.shape[2:]
		for bin_size in self.pyramids:
			x = F.adaptive_avg_pool2d(input, output_size=bin_size)
			x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
			feat  = feat + x
		return feat
    
class _ICHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ICHead, self).__init__()
        #self.cff_12 = CascadeFeatureFusion(512, 64, 128, nclass, norm_layer, **kwargs)
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer, **kwargs)
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, nclass, norm_layer, **kwargs)

        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        # x_cff_12, x_12_cls = self.cff_12(x_sub2, x_sub1)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True)
        outputs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()

        return outputs


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d, bias=False, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls



