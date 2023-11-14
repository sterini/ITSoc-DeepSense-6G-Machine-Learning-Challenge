# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 08:24:41 2023

@author: Umt
"""
import numpy as np
import pandas as pd
import os
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from math import sqrt
from PIL import Image
import torchvision.transforms as T

class DeepVerseChallengeLoaderTaskOne(Dataset):
    def __init__(self, csv_path, out_image_size=(256, 256)): 
        self.table = pd.read_csv(csv_path)
        self.dataset_folder = os.path.dirname(csv_path)
        self.position = self.table[['x', 'y']].to_numpy()
        self.images = self.table[['cam_left', 'cam_mid', 'cam_right']].to_numpy()
        self.out_image_size = out_image_size
        self.transform = T.Compose([T.PILToTensor(), T.Resize(size=self.out_image_size)])
        self.out_image_size = out_image_size
        
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        # Position
        P = torch.from_numpy(self.position[idx])
        
        # Channel
        ch_data_path = os.path.join(self.dataset_folder, self.table.loc[idx, 'channel'])
        H = torch.from_numpy(scipy.io.loadmat(ch_data_path)['channel'])

        # Return a single image or all three images from left/center/right cameras
        I = [self.transform(Image.open(self.images[idx, i])) for i in range(3)]
        
        # Radar
        radar_data_path = os.path.join(self.dataset_folder, self.table.loc[idx, 'radar'])
        R = torch.from_numpy(scipy.io.loadmat(radar_data_path)['ra'])
        
        return (P, R, *I), H

class DeepVerseChallengeLoaderTaskTwo(Dataset):
    def __init__(self, csv_path, out_image_size=(256, 256)): 
        self.table = pd.read_csv(csv_path)
        self.dataset_folder = os.path.dirname(csv_path)
        self.position = self.table[['x', 'y']].to_numpy()
        self.images = self.table[['cam_left', 'cam_mid', 'cam_right']].to_numpy()
        self.out_image_size = out_image_size
        self.transform = T.Compose([T.PILToTensor(), T.Resize(size=self.out_image_size)])
        self.out_image_size = out_image_size
        
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        # Position
        P = torch.from_numpy(self.position[idx])
        
        # Channel
        ch_data_path = os.path.join(self.dataset_folder, self.table.loc[idx, 'channel'])
        H = torch.from_numpy(scipy.io.loadmat(ch_data_path)['channel'])

        # Return a single image or all three images from left/center/right cameras
        I = [self.transform(Image.open(self.images[idx, i])) for i in range(3)]
        
        # Radar
        radar_data_path = os.path.join(self.dataset_folder, self.table.loc[idx, 'radar'])
        R = torch.from_numpy(scipy.io.loadmat(radar_data_path)['ra'])
        
        return (H, P, R, *I), H

class DeepVerseChallengeLoaderTaskThree(Dataset):
    def __init__(self, 
                 csv_path, 
                 out_image_size=(256, 256),
                 sequence_len=5
                 ): 
        self.table = pd.read_csv(csv_path)
        self.dataset_folder = os.path.dirname(csv_path)
        self.position = self.table[['x', 'y']].to_numpy()
        self.images = self.table[['cam_left', 'cam_mid', 'cam_right']].to_numpy()
        self.out_image_size = out_image_size
        self.transform = T.Compose([T.PILToTensor(), T.Resize(size=self.out_image_size)])
        
        self.sequence_len = sequence_len
        self.sequences = []
        user_idx = self.table['user_idx'].to_numpy()
        for i in range(user_idx.max()+1):
            user_samples = user_idx == i
            user_samples_idx = np.where(user_samples)[0]
            start = 0
            end = self.sequence_len
            while end <= len(user_samples_idx):
                self.sequences.append(user_samples_idx[start:end])
                start +=1
                end += 1
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        
        sample_idx = self.sequences[idx]
        sample_set = []
        for sample in sample_idx:
            # Position
            P = torch.from_numpy(self.position[sample])
            
            # Channel
            ch_data_path = os.path.join(self.dataset_folder, self.table.loc[sample, 'channel'])
            H = torch.from_numpy(scipy.io.loadmat(ch_data_path)['channel'])
                
            I = [self.transform(Image.open(self.images[sample, i])) for i in range(3)]
            
            # Radar
            radar_data_path = os.path.join(self.dataset_folder, self.table.loc[sample, 'radar'])
            R = torch.from_numpy(scipy.io.loadmat(radar_data_path)['ra'])
            
            sample_set.append([(H, P, R, *I), H])
        return sample_set

if __name__ == '__main__':

    # Task 1    
    # train_dataset = DeepVerseChallengeLoaderTaskOne(csv_path = r'./dataset_train.csv')
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # test_dataset = DeepVerseChallengeLoaderTaskOne(csv_path = r'./dataset_validation.csv')
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Task 2
    # train_dataset = DeepVerseChallengeLoaderTaskTwo(csv_path = r'./dataset_train.csv')
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # test_dataset = DeepVerseChallengeLoaderTaskTwo(csv_path = r'./dataset_validation.csv')
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Task 3
    train_dataset = DeepVerseChallengeLoaderTaskThree(csv_path = r'./dataset_train.csv')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    test_dataset = DeepVerseChallengeLoaderTaskThree(csv_path = r'./dataset_validation.csv')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)
