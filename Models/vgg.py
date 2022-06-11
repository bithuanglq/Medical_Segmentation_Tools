import torch
import torch.nn as nn
from torchsummary import summary



def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv3d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm3d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool3d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class VGG(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(512*5*6*4, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        VGG_features = self.layer5(out)
        out = VGG_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return VGG_features, out

'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1    [-1, 64, 160, 192, 128]           5,248
       BatchNorm3d-2    [-1, 64, 160, 192, 128]             128
              ReLU-3    [-1, 64, 160, 192, 128]               0
            Conv3d-4    [-1, 64, 160, 192, 128]         110,656
       BatchNorm3d-5    [-1, 64, 160, 192, 128]             128
              ReLU-6    [-1, 64, 160, 192, 128]               0
         MaxPool3d-7       [-1, 64, 80, 96, 64]               0
            Conv3d-8      [-1, 128, 80, 96, 64]         221,312
       BatchNorm3d-9      [-1, 128, 80, 96, 64]             256
             ReLU-10      [-1, 128, 80, 96, 64]               0
           Conv3d-11      [-1, 128, 80, 96, 64]         442,496
      BatchNorm3d-12      [-1, 128, 80, 96, 64]             256
             ReLU-13      [-1, 128, 80, 96, 64]               0
        MaxPool3d-14      [-1, 128, 40, 48, 32]               0
           Conv3d-15      [-1, 256, 40, 48, 32]         884,992
      BatchNorm3d-16      [-1, 256, 40, 48, 32]             512
             ReLU-17      [-1, 256, 40, 48, 32]               0
           Conv3d-18      [-1, 256, 40, 48, 32]       1,769,728
      BatchNorm3d-19      [-1, 256, 40, 48, 32]             512
             ReLU-20      [-1, 256, 40, 48, 32]               0
           Conv3d-21      [-1, 256, 40, 48, 32]       1,769,728
      BatchNorm3d-22      [-1, 256, 40, 48, 32]             512
             ReLU-23      [-1, 256, 40, 48, 32]               0
        MaxPool3d-24      [-1, 256, 20, 24, 16]               0
           Conv3d-25      [-1, 512, 20, 24, 16]       3,539,456
      BatchNorm3d-26      [-1, 512, 20, 24, 16]           1,024
             ReLU-27      [-1, 512, 20, 24, 16]               0
           Conv3d-28      [-1, 512, 20, 24, 16]       7,078,400
      BatchNorm3d-29      [-1, 512, 20, 24, 16]           1,024
             ReLU-30      [-1, 512, 20, 24, 16]               0
           Conv3d-31      [-1, 512, 20, 24, 16]       7,078,400
      BatchNorm3d-32      [-1, 512, 20, 24, 16]           1,024
             ReLU-33      [-1, 512, 20, 24, 16]               0
        MaxPool3d-34       [-1, 512, 10, 12, 8]               0
           Conv3d-35       [-1, 512, 10, 12, 8]       7,078,400
      BatchNorm3d-36       [-1, 512, 10, 12, 8]           1,024
             ReLU-37       [-1, 512, 10, 12, 8]               0
           Conv3d-38       [-1, 512, 10, 12, 8]       7,078,400
      BatchNorm3d-39       [-1, 512, 10, 12, 8]           1,024
             ReLU-40       [-1, 512, 10, 12, 8]               0
           Conv3d-41       [-1, 512, 10, 12, 8]       7,078,400
      BatchNorm3d-42       [-1, 512, 10, 12, 8]           1,024
             ReLU-43       [-1, 512, 10, 12, 8]               0
        MaxPool3d-44         [-1, 512, 5, 6, 4]               0
           Linear-45                 [-1, 4096]     251,662,336
      BatchNorm1d-46                 [-1, 4096]           8,192
             ReLU-47                 [-1, 4096]               0
           Linear-48                 [-1, 4096]      16,781,312
      BatchNorm1d-49                 [-1, 4096]           8,192
             ReLU-50                 [-1, 4096]               0
           Linear-51                    [-1, 2]           8,194
================================================================
Total params: 312,612,290
Trainable params: 312,612,290
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 45.00
Forward/backward pass size (MB): 16103.16
Params size (MB): 1192.52
Estimated Total Size (MB): 17340.68
----------------------------------------------------------------
'''

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VGG(n_classes=2).to(device)
    print(net)
    summary(net, (3, 160, 192 ,128))
