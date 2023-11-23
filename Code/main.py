import numpy as np
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select the GPU index

import sys
import warnings
from task1 import task1_train
from task2 import task2_train
from task3 import task3_train
from utils import *
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Parse input arguments
parser = argparse.ArgumentParser(description='Deepverse Challenge', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Task specific arguments
parser.add_argument('--TASK',type=str, default='task1', help='task name')
parser.add_argument('--GPS', type=str2bool, default=True, help='GPS')
parser.add_argument('--CAMERAS', type=str2bool, default=True, help='CAMERAS')
parser.add_argument('--RADAR', type=str2bool, default=True, help='RADAR')
parser.add_argument('--USE_PRESET', type=str2bool, default=False, help='USE_PRESET')
parser.add_argument('--f', type=str, default='task1.html', help='Html file of a plot')

# Training arguments
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--patience', type=int, default=15, help='patience')
parser.add_argument('--reduction', type=int, default=8, help='reduction')
parser.add_argument('--expansion', type=int, default=20, help='expansion')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--accumulation_steps', type=int, default=20, help='accumulation steps')

# Arguments related to sharing via email
parser.add_argument('--SHARE', type=str2bool, default=False, help='Do you want to share results via email [True/False]')

args = parser.parse_args()

def main(args):
    if args.SHARE:
        sender, receiver, sender_password = get_user_input()
    # Task 1
    if args.TASK == 'task1':
        train_losses,val_losses,avg_mse = task1_train(args)

    # Task 2    
    elif args.TASK == 'task2':
        train_losses,val_losses,avg_mse = task2_train(args)

    # Task 3
    elif args.TASK == 'task3':
        train_losses,val_losses,avg_mse = task3_train(args)

    else:
        raise ValueError('Invalid task name!')
    
    fig = gen_plot(args, train_losses, val_losses)
    
    if args.SHARE:
        send2email(args, fig, avg_mse, sender, receiver, sender_password, args.f)
    else:
        gen_html(fig, args.f)

    np.save(f'Results/train_loss_{args.f.split(".")[0]}.npy', train_losses)
    np.save(f'Results/val_loss_{args.f.split(".")[0]}.npy', val_losses)
    print('numpy array losses were saved to your current directory')

if __name__ == '__main__':
    main(args)
