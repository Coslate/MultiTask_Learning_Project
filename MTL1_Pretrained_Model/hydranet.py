import torch.nn as nn
import torch.nn.init as init


#Model Architecture
def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
    """3x3 Convolution: Depthwise:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias, groups=groups)

def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False,):
    "1x1 Convolution: Pointwise"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias, groups=groups)

def batchnorm(num_features):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """
    return nn.BatchNorm2d(num_features, affine=True, eps=1e-5, momentum=0.1)

def convbnrelu(in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels))

class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block from https://arxiv.org/abs/1801.04381"""
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super().__init__() # Python 3
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1) # Boolean/Condition
        self.output = nn.Sequential(convbnrelu(in_planes, intermed_planes, 1),
                                    convbnrelu(intermed_planes, intermed_planes, 3, stride=stride, groups=intermed_planes),
                                    convbnrelu(intermed_planes, out_planes, 1, act=False))

    def forward(self, x):
        #residual = x
        out = self.output(x)
        if self.residual:
            return (out + x)#+residual
        else:
            return out


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

class CRPBlock(nn.Module):
    """CRP definition"""
    def __init__(self, in_planes, out_planes, n_stages, groups=False):
        super().__init__()
        for i in range(n_stages):
            if groups:
                setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False, groups= in_planes if i==0 else out_planes))
            else:
                setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False, groups=1))

        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x

class HydraNet(nn.Module):
    """Net Definition"""
    def __init__(self, in_channels=32, num_classes=6, num_tasks=2, agg_size=256, n_crp=4):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
    ## Encoder-MobileNetV2 ##
        self.mobilenet_config = [[1, 16, 1, 1], # expansion rate, output channels, number of repeats, stride
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                    ]
        self.in_channels = in_channels # number of input channels
        self.num_layers = len(self.mobilenet_config)
        self.layer1 = convbnrelu(3, self.in_channels, kernel_size=3, stride=2)

        c_layer = 2
        for t,c,n,s in (self.mobilenet_config):
            layers = []
            for idx in range(n):
                layers.append(InvertedResidualBlock(self.in_channels, c, expansion_factor=t, stride=s if idx == 0 else 1))
                self.in_channels = c
            setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers)) # setattr(object, name, value)
            c_layer += 1

        ## 🔹 Add Expansion Layer (320 → 1280 Channels) ##
        #self.final_conv = conv1x1(320, 1280, bias=False)  # Expand channels

        ## Decoder-Light-Weight RefineNet ##
        #self.conv8 = conv1x1(1280, 256, bias=False)  # Updated input channels to 1280            
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
        self.crp4 = self._make_crp(in_planes=256, out_planes=256, stages=4, groups=False)
        self.crp3 = self._make_crp(in_planes=256, out_planes=256, stages=4, groups=False)
        self.crp2 = self._make_crp(in_planes=256, out_planes=256, stages=4, groups=False)
        self.crp1 = self._make_crp(in_planes=256, out_planes=256, stages=4, groups=True)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.pre_depth = conv1x1(256, 256, groups=256, bias=False)# Define the Purple Pre-Head for Depth
        self.depth = conv3x3(256, 1, bias=True)# Define the Final layer of Depth
        self.pre_segm = conv1x1(256, 256, groups=256, bias=False)#: Call the Purple Pre-Head for Segm
        self.segm = conv3x3(256, self.num_classes, bias=True)#: Define the Final layer of Segmentation
        self.relu = nn.ReLU6(inplace=True)#: Define a RELU 6 Operation

        if self.num_tasks == 3:
            # Create a Normal Head
            self.pre_normal = conv1x1(256, 256, groups=256, bias=False)
            self.normal = conv3x3(256, 3, bias=True)

    @staticmethod
    def _initialize_weights(m):
        """
        Apply weight initialization based on the type of layer.
        """
        if isinstance(m, nn.Conv2d):
            # Kaiming (He) Initialization for Conv layers
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            # Xavier Initialization for Fully Connected Layers
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            # BatchNorm Initialization (mean=1, variance=0)
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)              

    def extract_encoder(self):
        """ Returns only the encoder part of the HydraNet """
        encoder_layers = [
            self.layer1,
            self.layer2, 
            self.layer3, 
            self.layer4, 
            self.layer5, 
            self.layer6, 
            self.layer7, 
            self.layer8
            #self.final_conv  # 🔹 Include the new expansion layer (320 → 1280)
        ]
        self.apply(self._initialize_weights)
        return nn.Sequential(*encoder_layers)            

    def forward(self, x):
        # MOBILENET V2
        x = self.layer1(x)
        x = self.layer2(x) # x / 2
        l3 = self.layer3(x) # 24, x / 4
        l4 = self.layer4(l3) # 32, x / 8
        l5 = self.layer5(l4) # 64, x / 16
        l6 = self.layer6(l5) # 96, x / 16
        l7 = self.layer7(l6) # 160, x / 32
        l8 = self.layer8(l7) # 320, x / 32
        ## 🔹 Apply Final Expansion Layer (320 → 1280 Channels)
        #l8 = self.final_conv(l8)        

        # LIGHT-WEIGHT REFINENET
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size=l6.size()[2:], mode='bilinear', align_corners=False)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size=l4.size()[2:], mode='bilinear', align_corners=False)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=False)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        # HEADS 
        out_segm = self.pre_segm(l3)
        out_segm = self.relu(out_segm)
        out_segm = self.segm(out_segm)

        out_d = self.pre_depth(l3)
        out_d = self.relu(out_d)
        out_d = self.depth(out_d)

        if self.num_tasks == 3:
            out_n = self.pre_normal(l3)
            out_n = self.relu(out_n)
            out_n = self.normal(out_n)
            return out_segm, out_d, out_n
        else:
            return out_segm, out_d

    def _make_crp(self, in_planes, out_planes, stages, groups=False):
        layers = [CRPBlock(in_planes, out_planes,stages, groups=groups)]
        return nn.Sequential(*layers)