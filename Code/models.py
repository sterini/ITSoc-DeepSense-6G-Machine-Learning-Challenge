#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import math
from collections import OrderedDict
from utils import CSI_back2original,CSI_reshape,decimal_to_binary,binary_to_decimal
from data_processing_layers import *

################################################ TASK 1 ################################################

class task1decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the layers for processing GPS position
        self.gp = gpsdata()
        
        
        # Define the layers for processing radar data
        self.rd = radardata()
        
        # Define the layers for processing left camera image
        self.lc = leftcamera()
        
        # Define the layers for processing center camera image
        self.cc = centercamera()
        
        # Define the layers for processing right camera image
        self.rc = rightcamera()
        
        # Define the layers for final output
        self.output_fc = nn.Linear(16*5, 128 * 64)
        self.output_sig = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, gps, radar, left, center, right, onoffdict):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bs = gps.size(0)
        # Process GPS position
        if onoffdict['GPS']:
            gps_out = self.gp(gps)
        else:
            gps_out = torch.zeros(bs, 16).to(device)
        
        # Process radar data
        if onoffdict['RADAR']:
            radar_out = self.rd(radar)
        else:
            radar_out = torch.zeros(bs, 16).to(device)
        
        if onoffdict['CAMERAS']:
            # Process left camera image
            left_out = self.lc(left)

            # Process center camera image
            center_out = self.cc(center)

            # Process right camera image
            right_out = self.rc(right)
        else:
            left_out = torch.zeros(bs, 16).to(device)

            # Process center camera image
            center_out = torch.zeros(bs, 16).to(device)

            # Process right camera image
            right_out = torch.zeros(bs, 16).to(device)
        
        # Combining
        combined = torch.cat((gps_out, radar_out, left_out, center_out, right_out), dim=1)

        # Generate final output
        output = self.output_fc(combined)
        output = self.output_sig(output)
        output = output.view(output.size(0), 2, 64, 64)

        # Modify output based on mode
        
        output = CSI_back2original(output)

        return output
    
##################################################################################################

class ACRDecoderBlock(nn.Module):
    r""" Inverted residual with extensible width and group conv
    """
    def __init__(self, expansion):
        super(ACRDecoderBlock, self).__init__()
        width = 8 * expansion
        self.conv1_bn = ConvBN(2, width, [1, 9])
        self.prelu1 = nn.PReLU(num_parameters=width, init=0.3)
        self.conv2_bn = ConvBN(width, width, 7, groups=4 * expansion)
        self.prelu2 = nn.PReLU(num_parameters=width, init=0.3)
        self.conv3_bn = ConvBN(width, 2, [9, 1])
        self.prelu3 = nn.PReLU(num_parameters=2, init=0.3)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        residual = self.prelu1(self.conv1_bn(x))
        residual = self.prelu2(self.conv2_bn(residual))
        residual = self.conv3_bn(residual)

        return self.prelu3(identity + residual)

class ACREncoderBlock(nn.Module):
    def __init__(self):
        super(ACREncoderBlock, self).__init__()
        self.conv_bn1 = ConvBN(2, 2, [1, 9])
        self.prelu1 = nn.PReLU(num_parameters=2, init=0.3)
        self.conv_bn2 = ConvBN(2, 2, [9, 1])
        self.prelu2 = nn.PReLU(num_parameters=2, init=0.3)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)
        residual = self.prelu1(self.conv_bn1(x))
        residual = self.conv_bn2(residual)
        return self.prelu2(identity + residual)

################################################ TASK 2 ################################################


class task2Encoder(nn.Module):
    
    def __init__(self, reduction=16, expansion=20):
        super(task2Encoder, self).__init__()
        n1=int(math.log2(reduction))
        self.encoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
            ("ACREncoderBlock1", ACREncoderBlock()),
            ("ACREncoderBlock2", ACREncoderBlock()),
        ]))
        
        
        conv_layers = []
        for n in range(n1):
            conv = nn.Conv2d(2**(n + 1), 2**(n + 2), kernel_size=3, stride=2, padding=1)
            conv_layers.append(conv)
            conv_layers.append(nn.PReLU(num_parameters=2**(n + 2), init=0.3))
        self.convn = nn.Sequential(*conv_layers)
        self.output_sig = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, h, w = x.detach().size()
        x= CSI_reshape(x)
        out = self.encoder_feature(x.to(torch.float32))
        out = self.convn(out)
        out = self.output_sig(out)
        #print(out.shape)
        return out.reshape(n,1,-1).to(torch.float16)
       


class task2Decoder(nn.Module):
    
    def __init__(self, reduction=16, expansion=20):
        super(task2Decoder, self).__init__()
        self.total_size = 8192
        w, h =64, 64
        self.reduced_size = self.total_size//reduction
        
        self.n1=int(math.log2(reduction))
        self.wo =w//(2**(self.n1+1))
        self.ho =h//(2**(self.n1+1))
        # Define the layers for processing GPS position
        self.gp = gpsdata()     
        
        # Define the layers for processing radar data
        self.rd = radardata()
        
        # Define the layers for processing left camera image
        self.lc = leftcamera()
        
        # Define the layers for processing center camera image
        self.cc = centercamera()
        
        # Define the layers for processing right camera image
        self.rc = rightcamera()
        
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
            ("ACRDecoderBlock1", ACRDecoderBlock(expansion=expansion)),
            ("ACRDecoderBlock2", ACRDecoderBlock(expansion=expansion)),
            ("sigmoid", nn.Sigmoid())
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
        self.sigmoid = nn.Sigmoid()
        self.lr1=nn.LeakyReLU(negative_slope=0.3, inplace=True)
        
        # Define the layers for final output
        self.output_fc1 = nn.Linear(self.reduced_size+16*5, self.reduced_size)
        self.output_fc2 = nn.Linear(self.reduced_size, self.reduced_size)
        
        #self.output_sig = nn.Sigmoid()
        convtrans_layers = []
        for n in range(self.n1):
            convtrans = nn.ConvTranspose2d(2**(self.n1-n+1), 2**(self.n1-n), kernel_size=3, stride=2, padding=1, output_padding=1)
            convtrans_layers.append(convtrans)
            convtrans_layers.append(nn.PReLU(num_parameters=2**(self.n1-n), init=0.3))
        self.convntrans = nn.Sequential(*convtrans_layers)
         
    
    def forward(self, Hencoded, gps, radar, left, center, right, onoffdict):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bs = Hencoded.size(0)
        
        # Process GPS position
        if onoffdict['GPS']:
            gps_out = self.gp(gps)
        else:
            gps_out = torch.zeros(bs, 16).to(device)
        
        # Process radar data
        if onoffdict['RADAR']:
            radar_out = self.rd(radar)
        else:
            radar_out = torch.zeros(bs, 16).to(device)
        
        if onoffdict['CAMERAS']:
            # Process left camera image
            left_out = self.lc(left)

            # Process center camera image
            center_out = self.cc(center)

            # Process right camera image
            right_out = self.rc(right)
        else:
            left_out = torch.zeros(bs, 16).to(device)

            # Process center camera image
            center_out = torch.zeros(bs, 16).to(device)

            # Process right camera image
            right_out = torch.zeros(bs, 16).to(device)
        
        #combining
        combined = torch.cat((Hencoded.view(-1,self.reduced_size), gps_out, radar_out, left_out, center_out, right_out), dim=1)
        
        # Generate final output
        output = self.lr1(self.output_fc1(combined))
        output = self.lr1(self.output_fc2(output))
        
        output = self.convntrans(output.view(-1,2**(self.n1+1),self.wo,self.ho).to(torch.float32))
        output = self.decoder_feature(output)
        output = self.sigmoid(output)
        output = output.view(bs,2, 64, 64)
        output = CSI_back2original(output)
        return output
       

class task2model(nn.Module):
    def __init__(self, reduction=16, expansion=20):
        super().__init__()
        
        self.en=task2Encoder(reduction)
        
        self.de=task2Decoder(reduction,expansion)
   
    def forward(self, Hin, gps, radar, left, center, right, device, is_training, onoffdict): 
        
        #Encoder
        Hencoded=self.en(Hin)
        
        if not is_training:
            #convert to 64 bit binary at transmitter
            binary_representation = decimal_to_binary(Hencoded, device)
            
            # At receiver convert back to decimal
            Hreceived= binary_to_decimal(binary_representation, device)
        else:
            Hreceived=Hencoded
        #Decoder   
        Hdecoded=self.de(Hreceived, gps, radar, left, center, right, onoffdict)
        

        return Hdecoded
    
################################################ TASK 3 ################################################

class task3Encoder(nn.Module):
    
    def __init__(self, reduction=16, expansion=20):
        super(task3Encoder, self).__init__()
        n1=int(math.log2(reduction))
        self.encoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
            ("ACREncoderBlock1", ACREncoderBlock()),
            ("ACREncoderBlock2", ACREncoderBlock()),
        ]))
       
        conv_layers = []
        for n in range(n1):
            conv = nn.Conv2d(2**(n + 1), 2**(n + 2), kernel_size=3, stride=2, padding=1)
            conv_layers.append(conv)
            conv_layers.append(nn.PReLU(num_parameters=2**(n + 2), init=0.3))
        self.convn = nn.Sequential(*conv_layers)
        self.output_sig = nn.Sigmoid()
        
        self.a = nn.Parameter(nn.init.xavier_uniform_(torch.empty((1, 512))))
        self.b = nn.Parameter(nn.init.xavier_uniform_(torch.empty((1, 512))))
        self.c = nn.Parameter(nn.init.xavier_uniform_(torch.empty((1, 512))))
        self.d = nn.Parameter(nn.init.xavier_uniform_(torch.empty((1, 512))))
        
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, input_autoregressive_features):
        
        # Reshape the parameters to match the batch size
        bs = x.size(0)
        a = self.a.expand(bs, -1)
        b = self.b.expand(bs, -1)
        c = self.c.expand(bs, -1)
        d = self.d.expand(bs, -1)
        
        out_tminus1 = input_autoregressive_features[:,0,:].view(-1,512)
        out_tminus2 = input_autoregressive_features[:,1,:].view(-1,512)
        x = CSI_reshape(x)
        out = self.encoder_feature(x.to(torch.float32))
        out = self.convn(out)
        out = self.output_sig(out)
        out = out.reshape(bs,-1)
        
        out=out*a+out_tminus1*b+out_tminus2*c+d
        
        
        out=self.output_sig(out).view(bs,1,-1)
        out=out.to(torch.float16)
        encoded_features=out
        autoregressive_features=out
        return encoded_features, autoregressive_features


class task3Decoder(nn.Module):
    
    def __init__(self, reduction=16, expansion=20):
        super(task3Decoder, self).__init__()
        self.total_size = 8192
        w, h =64, 64
        self.reduced_size = self.total_size//reduction
        
        self.n1=int(math.log2(reduction))
        self.wo =w//(2**(self.n1+1))
        self.ho =h//(2**(self.n1+1))
        
        # Define the layers for processing GPS position
        self.gp = gpsdata()     
        
        # Define the layers for processing radar data
        self.rd = radardata()
        
        # Define the layers for processing left camera image
        self.lc = leftcamera()
        
        # Define the layers for processing center camera image
        self.cc = centercamera()
        
        # Define the layers for processing right camera image
        self.rc = rightcamera()
        
       
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
            ("ACRDecoderBlock1", ACRDecoderBlock(expansion=expansion)),
            ("ACRDecoderBlock2", ACRDecoderBlock(expansion=expansion)),
            ("sigmoid", nn.Sigmoid())
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        self.lr1=nn.LeakyReLU(negative_slope=0.3, inplace=True)
        
        # Define the layers for final output
        self.output_fc1 = nn.Linear(self.reduced_size+16*5+self.reduced_size*2, self.reduced_size)
        self.output_fc2 = nn.Linear(self.reduced_size, self.reduced_size)
        convtrans_layers = []
        for n in range(self.n1):
            convtrans = nn.ConvTranspose2d(2**(self.n1-n+1), 2**(self.n1-n), kernel_size=3, stride=2, padding=1, output_padding=1)
            convtrans_layers.append(convtrans)
            convtrans_layers.append(nn.PReLU(num_parameters=2**(self.n1-n), init=0.3))
        self.convntrans = nn.Sequential(*convtrans_layers)
        
        #Layers for auto regression 
        self.a= nn.Parameter(torch.randn(self.reduced_size))
        self.b= nn.Parameter(torch.randn(self.reduced_size))
        self.c= nn.Parameter(torch.randn(self.reduced_size))
        self.d= nn.Parameter(torch.randn(self.reduced_size))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, Hencoded, input_autoregressive_features, gps, radar, left, center, right, onoffdict):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bs = Hencoded.size(0)
        out_tminus1=input_autoregressive_features[:,0,:].view(bs,-1)
        out_tminus2=input_autoregressive_features[:,1,:].view(bs,-1)
    
        
        # Process GPS position
        if onoffdict['GPS']:
            gps_out = self.gp(gps)
        else:
            gps_out = torch.zeros(bs, 16).to(device)
        
        # Process radar data
        if onoffdict['RADAR']:
            radar_out = self.rd(radar)
        else:
            radar_out = torch.zeros(bs, 16).to(device)
        
        if onoffdict['CAMERAS']:
            # Process left camera image
            left_out = self.lc(left)

            # Process center camera image
            center_out = self.cc(center)

            # Process right camera image
            right_out = self.rc(right)
        else:
            left_out = torch.zeros(bs, 16).to(device)

            # Process center camera image
            center_out = torch.zeros(bs, 16).to(device)

            # Process right camera image
            right_out = torch.zeros(bs, 16).to(device)
        
        
        #combining
        combined = torch.cat((Hencoded.view(bs,-1), out_tminus1,  out_tminus2, gps_out, radar_out, left_out, center_out, right_out), dim=1)
        
        # Generate final output
        output = self.lr1(self.output_fc1(combined))
        output = self.lr1(self.output_fc2(output))
        output = self.convntrans(output.view(-1,2**(self.n1+1),self.wo,self.ho).to(torch.float32))
        output = self.decoder_feature(output)
        output = self.sigmoid(output)
        output = output.view(bs,2, 64, 64)
        output = CSI_back2original(output)
        return output


#complete task 3 model including encoder, decoder and channel
class task3model(nn.Module):
    def __init__(self, reduction=16, expansion=20):
        super().__init__()
        self.total_size = 8192
        self.reduced_size = self.total_size//reduction
        self.en = task3Encoder(reduction)
        self.de = task3Decoder(reduction, expansion)
        self.ar = [None] * 5  # List to store the AR variables
    
    
    def forward(self, X, time_index, device, is_training, onoffdict): 
         
        Hin = X[0].to(device)
        gps = X[1].to(device)
        radar = X[2].to(device)
        left = X[3].to(device)
        center = X[4].to(device)
        right = X[5].to(device)
        batch_size = Hin.shape[0]
        
        # Encoder
        if time_index == 0:
            iarf = torch.zeros((batch_size, 2, self.reduced_size), dtype=torch.float).to(device)
            Hencoded, self.ar[0] = self.en(Hin, iarf)
    
        elif time_index==1:
            iarf=torch.cat([self.ar[0].detach(), torch.zeros((batch_size, 1, self.reduced_size), dtype=torch.float).to(device)], dim=1)
            Hencoded, self.ar[1] =self.en(Hin, iarf)
            
        else:
            iarf = torch.cat([self.ar[time_index-1].detach(), self.ar[time_index-2].detach()], dim=1)
            Hencoded, self.ar[time_index] = self.en(Hin, iarf)
        
            
        if not is_training:
            # Convert to 64-bit binary at the transmitter
            binary_representation = decimal_to_binary(Hencoded, device)
            # At the receiver, convert back to decimal
            Hreceived = binary_to_decimal(binary_representation, device)
        else:
            Hreceived = Hencoded

        # Decoder   
        Hdecoded = self.de(Hencoded, iarf, gps, radar, left, center, right, onoffdict)

        return Hdecoded


