import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class ImageEncoder(torch.nn.Module):
    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 2, 2), (1, 1, 1))
        self.conv4 = nn.Conv3d(96, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1))

        #Recurrecnt layers
        self.gru1  = nn.GRU(128*6*8, 256, 1, bidirectional=True) # images
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)

        #Fully connected layers
        self.FC1   = nn.Linear(512, 64) 
        self.FC2   = nn.Linear(64*opt.num_frames, 64)

        #Activation function
        self.relu = nn.ReLU(inplace=True)
    
        #Dropout
        self.dropout = nn.Dropout(opt.drop)
        self.dropout3d = nn.Dropout3d(opt.drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
 
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout3d(x)
 
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
 
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
 
        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        x = self.FC1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(1, 0, 2).contiguous()
        x = x.view(x.size(0), -1)
        x = self.FC2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class AudioEncoder(torch.nn.Module):
    def __init__(self, opt):
        super(AudioEncoder, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 2, 2), (1, 1, 1))
        self.conv4 = nn.Conv3d(96, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1))

        #Recurrecnt layers
        self.gru1  = nn.GRU(128*8, 256, 1, bidirectional=True) # audio
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)

        #Fully connected layers
        self.FC1   = nn.Linear(512, 64) 
        self.FC2   = nn.Linear(64*(int(opt.segment_len*opt.sample_rate)//200+1), 64)    # audio
        #200 is the length of hop between STFT windows

        #Activation function
        self.relu = nn.ReLU(inplace=True)
    
        #Dropout
        self.dropout = nn.Dropout(opt.drop)
        self.dropout3d = nn.Dropout3d(opt.drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
 
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout3d(x)
 
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
 
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
 
        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)
 
        x = self.FC1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(1, 0, 2).contiguous()
        x = x.view(x.size(0), -1)
        x = self.FC2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SFNet(torch.nn.Module):
    def __init__(self, opt):
        super(SFNet, self).__init__()
        self.opt = opt

        if opt.mode in [1,4,5,7]:
            self.rgb_enc = ImageEncoder(self.opt)
        if opt.mode in [2,4,6,7]:
            self.thr_enc = ImageEncoder(self.opt)
        if opt.mode in [3,5,6,7]:
            self.aud_enc = AudioEncoder(self.opt)

        if opt.predict.lower() == "gender":
            if opt.mode in [1,2,3]:
                self.FC1 = nn.Linear(64, 1)
            elif opt.mode in [4,5,6]:
                self.FC1 = nn.Linear(64*2, 1)
            elif opt.mode in [7]:
                self.FC1 = nn.Linear(64*3, 1)
        elif opt.predict.lower() == "age":
            if opt.mode in [1,2,3]:
                self.FC1 = nn.Linear(64, 4)
            elif opt.mode in [4,5,6]:
                self.FC1 = nn.Linear(64*2, 4)
            elif opt.mode in [7]:
                self.FC1 = nn.Linear(64*3, 4)
        else:
            print("Incorrect prediction value is given! TERMINATING...")
            exit()

    def forward(self, x1=None, x2=None, x3=None):
        #x1-rgb, x2-thremal, x3=audio
        if self.opt.mode in [1,4,5,7]:
            x1 = self.rgb_enc(x1)
        if self.opt.mode in [2,4,6,7]:
            x2 = self.thr_enc(x2)
        if self.opt.mode in [3,5,6,7]:
            x3 = self.aud_enc(x3)

        if self.opt.mode == 1:
            x  = self.FC1(x1)
        elif self.opt.mode == 2:
            x  = self.FC1(x2)
        elif self.opt.mode == 3:
            x  = self.FC1(x3)
        elif self.opt.mode == 4:
            x  = self.FC1(torch.cat((x1,x2),1))
        elif self.opt.mode == 5:
            x  = self.FC1(torch.cat((x1,x3),1))
        elif self.opt.mode == 6:
            x  = self.FC1(torch.cat((x2,x3),1))
        elif self.opt.mode == 7:
            x  = self.FC1(torch.cat((x1,x2,x3),1))
        return x
