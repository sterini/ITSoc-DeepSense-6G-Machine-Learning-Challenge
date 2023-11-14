import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Select the GPU index
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import warnings
from torch_challenge_dataset import DeepVerseChallengeLoaderTaskThree
from models import task3model
from utils import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def task3_train(args):
    #Parameters
    onoffdict= {'GPS': args.GPS, 'CAMERAS': args.CAMERAS, 'RADAR': args.RADAR}
    if args.USE_PRESET == True:
        lr=0.001
        num_epochs=100
        patience=15
        reduction=8 
        expansion=20
        batch_size = 200
        accumulation_steps = 20
    else:
        lr=args.lr
        num_epochs=args.num_epochs
        patience=args.patience
        reduction=args.reduction
        expansion=args.expansion
        batch_size=args.batch_size
        accumulation_steps = args.accumulation_steps
    
    weight_path=f'models/task3/cr{reduction}/exp{expansion}/gps{onoffdict["GPS"]}_cam{onoffdict["CAMERAS"]}_rad{onoffdict["RADAR"]}/'


    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    
    effective_batch_size = batch_size // accumulation_steps  # Calculate the effective batch size

    # Task 3
    train_dataset = DeepVerseChallengeLoaderTaskThree(csv_path = r'./dataset_train.csv')
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=24)
    test_dataset = DeepVerseChallengeLoaderTaskThree(csv_path =  r'./dataset_validation.csv')
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=24)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model= task3model()
    print(f'Number of parameters in Task3 Encoder: {cal_model_parameters(model.en)}')
    print(f'Number of parameters in Task3 Decoder: {cal_model_parameters(model.de)}')

    ##### Training #####

    # Check if "models" folder exists, create it if it doesn't
    if not os.path.exists("models"):
        os.makedirs("models")

    # Loss function
    criterion= nn.MSELoss().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Scheduler
    scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                                T_max=num_epochs ,
                                                T_warmup=5 ,
                                                eta_min=1e-6)

    model = model.to(device)

    start_time = time.time()
    num_train_batches=len(train_loader)
    num_test_batches=len(test_loader)
    train_losses = []
    val_losses = []
    patience_counter = 0
    best_val_loss = float('inf')

    for i in range(num_epochs):
        loss1 = 0
        epoch_time = time.time()
        model.train()
        optimizer.zero_grad()
        # Run the training batches
        
        for b, t_x in enumerate(train_loader):
            model.ar = [None] * 5  
            for time_index, (X, y) in enumerate(t_x):
                y_train = y.to(device)
                y_train_reshaped = CSI_reshape(y_train)
                
                # Get the input and output for the given time index
                y_pred = model(X, time_index, device, is_training=True, onoffdict = onoffdict)
                y_pred = CSI_reshape(y_pred)
                
                loss = criterion(y_pred, y_train_reshaped)
                loss = loss / accumulation_steps  # Scale the loss
                loss1 += loss.item()
                loss.backward()
                
            if (b + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        train_loss = loss1 / (num_train_batches * 5)
        train_losses.append(train_loss)
        
        # Update the learning rate scheduler
        scheduler.step()
        
        # Run the testing batches
        model.eval()
        with torch.no_grad():
            loss1 = 0
            for b, t_x in enumerate(test_loader):
                model.ar = [None] * 5 
                for time_index, (X, y) in enumerate(t_x):
                    y_test = y.to(device)
                    y_test_reshaped = CSI_reshape(y_test)
                    
                    # Get the input and output for the given time index
                    y_pred = model(X, time_index, device, is_training=True, onoffdict = onoffdict)
                    y_pred = CSI_reshape(y_pred)
                    
                    loss = criterion(y_pred, y_test_reshaped)
                    loss=loss/accumulation_steps
                    loss1 += loss.item()
            
            val_loss = loss1 / (num_test_batches * 5)
            val_losses.append(val_loss)
        
        
        print(f'epoch:{i+1}/{num_epochs} average TL:{train_loss:.8f} average VL:{val_loss:.8f} epoch time:{time.time() - epoch_time:.0f} seconds, lr:{optimizer.param_groups[0]["lr"]:.2e}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, weight_path + "task3.pth")
            torch.save(model.en, weight_path + "task3Encoder.pth")
            torch.save(model.de, weight_path + "task3Decoder.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Validation loss did not decrease for {patience} epochs. Stopping training.')
            break  
            
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed            


    ##### Inference #####

    model2=torch.load(weight_path + "task3.pth").to(device)
    # Run the testing batches
    model2.eval()
    with torch.no_grad():
        mse1=0
        for b,t_x in enumerate(test_loader):
            model.ar = [None] * 5 
            for time_index,(X, y) in enumerate(t_x):
                y_test=y.to(device)
                #get the input and output for the given time index
                #X, y = t_x[time_index]
                y_pred=model2(X, time_index, device, is_training=False, onoffdict = onoffdict)
                y_test_reshaped=CSI_reshape(y_test)
                y_pred=CSI_reshape(y_pred)
                mse0 = criterion(y_pred, y_test_reshaped) 
                mse1+=mse0         
        avg_mse=mse1/(5*num_test_batches)
    return train_losses, val_losses, avg_mse






