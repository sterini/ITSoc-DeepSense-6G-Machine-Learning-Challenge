import torch
import torch.nn as nn
from collections import OrderedDict
from utils import normalize_image

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class radardata(nn.Module):
    
    def __init__(self):
        super(radardata, self).__init__()
       
        self.conv1_bn = nn.BatchNorm2d(1)  # BatchNorm2d layer with 1 channel
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(1, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(1, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        
        self.conv1 = nn.Conv2d(2, 6, kernel_size=4, stride=2, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 2, kernel_size=4, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lr1=nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.encoder_fc = nn.Linear(2*32*8,16)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x=x.view(-1,1,512,128)
        x = self.conv1_bn(x)
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        #print(out.shape)
        out = self.lr1(self.conv1(out))
        out=self.pool1(out)
        out = self.lr1(self.conv2(out))
        out = self.pool2(out)
        out = self.lr1(self.encoder_fc(out.view(x.size(0), -1)))
        return out 


class leftcamera(nn.Module):
    
    def __init__(self):
        super(leftcamera, self).__init__()
       
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(3, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(3, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        
        self.conv1 = nn.Conv2d(2, 6, kernel_size=4, stride=2, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 2, kernel_size=4, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lr1=nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.encoder_fc = nn.Linear(2*16*16,16)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = normalize_image(x).to(torch.float32)
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        #print(out.shape)
        out = self.lr1(self.conv1(out))
        out=self.pool1(out)
        out = self.lr1(self.conv2(out))
        out = self.pool2(out)
        out = self.lr1(self.encoder_fc(out.view(x.size(0), -1)))
        
        return out
       

class centercamera(nn.Module):
    
    def __init__(self):
        super(centercamera, self).__init__()
        
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(3, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(3, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        
        self.conv1 = nn.Conv2d(2, 6, kernel_size=4, stride=2, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 2, kernel_size=4, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lr1=nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.encoder_fc = nn.Linear(2*16*16,16)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
                
    def forward(self, x):
        
        x= normalize_image(x).to(torch.float32)
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.lr1(self.conv1(out))
        out=self.pool1(out)
        out = self.lr1(self.conv2(out))
        out = self.pool2(out)
        out = self.lr1(self.encoder_fc(out.view(x.size(0), -1)))
        
        return out


class rightcamera(nn.Module):
    
    def __init__(self):
        super(rightcamera, self).__init__()
        
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(3, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(3, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        
        self.conv1 = nn.Conv2d(2, 6, kernel_size=4, stride=2, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 2, kernel_size=4, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lr1=nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.encoder_fc = nn.Linear(2*16*16,16)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x= normalize_image(x).to(torch.float32)
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.lr1(self.conv1(out))
        out=self.pool1(out)
        out = self.lr1(self.conv2(out))
        out = self.pool2(out)
        out = self.lr1(self.encoder_fc(out.view(x.size(0), -1)))
        
        return out


class gpsdata(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers for processing GPS position
        self.gps_bn = nn.BatchNorm1d(2)  # BatchNorm1d layer with 2 channels
        self.gps_fc = nn.Linear(2, 16)
        
        self.gps_relu = nn.ReLU()
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.xavier_uniform_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, gps):  
        # Process GPS position
        gps_out = self.gps_bn(gps.to(torch.float32)) # float64 to float32
        gps_out = self.gps_fc(gps_out)  
        gps_out = self.gps_relu(gps_out)
        return gps_out