import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select the GPU index
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import warnings
from torch_challenge_dataset import DeepVerseChallengeLoaderTaskTwo
from models import task2model
from utils import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def task2_train(args):


    #Parameters
    onoffdict= {'GPS': args.GPS, 'CAMERAS': args.CAMERAS, 'RADAR': args.RADAR}
    if args.USE_PRESET == True:
        lr=0.001
        num_epochs=100
        patience=15
        reduction=8 
        expansion=20
        batch_size = 100
    else:
        lr=args.lr
        num_epochs=args.num_epochs
        patience=args.patience
        reduction=args.reduction
        expansion=args.expansion
        batch_size=args.batch_size

    weight_path=f'models/task2/cr{reduction}/exp{expansion}/gps{onoffdict["GPS"]}_cam{onoffdict["CAMERAS"]}_rad{onoffdict["RADAR"]}/'
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    train_dataset = DeepVerseChallengeLoaderTaskTwo(csv_path = r'./dataset_train.csv')
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=24)
    test_dataset = DeepVerseChallengeLoaderTaskTwo(csv_path =  r'./dataset_validation.csv')
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, num_workers=24)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model=task2model()
    print(f'Number of parameters in Task2 Encoder: {cal_model_parameters(model.en)}')
    print(f'Number of parameters in Task2 Decoder: {cal_model_parameters(model.de)}')

    ###### Training ######

    # Loss function
    criterion= nn.MSELoss().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Scheduler
    scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                                T_max=num_epochs * len(train_loader),
                                                T_warmup=5 * len(train_loader),
                                                eta_min=1e-6)


    model=model.to(device)

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
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):  
            optimizer.zero_grad()
            y_train=y_train.to(device)
            # Apply the model
            y_pred=model(X_train[0].to(device),X_train[1].to(device),X_train[2].to(device),X_train[3].to(device),X_train[4].to(device),X_train[5].to(device),device, is_training=True, onoffdict = onoffdict)
            y_train_reshaped=CSI_reshape(y_train)
            y_pred=CSI_reshape(y_pred)
            loss = criterion(y_pred, y_train_reshaped) 
            # Update parameters
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss1=loss1+loss
        train_loss=loss1/num_train_batches  
        train_losses.append(train_loss.item())
        # Update the learning rate scheduler
    
        # Run the testing batches
        model.eval()
        with torch.no_grad():
            loss1=0
            for b, (X_test, y_test) in enumerate(test_loader):
                y_test=y_test.to(device)
                # Apply the model
                y_pred=model(X_test[0].to(device),X_test[1].to(device),X_test[2].to(device),X_test[3].to(device),X_test[4].to(device),X_test[5].to(device),device, is_training=True, onoffdict= onoffdict)
                y_test_reshaped=CSI_reshape(y_test)
                y_pred=CSI_reshape(y_pred)
                loss = criterion(y_pred, y_test_reshaped) 
                loss1=loss1+loss         
            val_loss=loss1/num_test_batches  
            val_losses.append(val_loss.item())
        
        print(f'epoch:{i+1}/{num_epochs} average TL:{train_loss.item():10.8f} average VL:{val_loss.item():10.8f} epoch time:{time.time() - epoch_time:.0f} seconds, lr:{optimizer.param_groups[0]["lr"]:.2e}')               
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, weight_path+"task2.pth")
            torch.save(model.en, weight_path+"task2Encoder.pth")
            torch.save(model.de, weight_path+"task2Decoder.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Validation loss did not decrease for {patience} epochs. Stopping training.')
            break  
            
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed            


    ##### Inference #####

    model2=torch.load(weight_path+"task2.pth").to(device)
    # Run the testing batches
    model2.eval()
    with torch.no_grad():
        mse1=0
        for b, (X_test, y_test) in enumerate(test_loader):

            y_test=y_test.to(device)
            # Apply the model
            y_pred=model2(X_test[0].to(device),X_test[1].to(device), X_test[2].to(device), X_test[3].to(device), X_test[4].to(device), X_test[5].to(device),device, is_training=False, onoffdict= onoffdict)
            #y_test_reshaped=CSI_reshape(y_test.to(device))
            y_test_reshaped=CSI_reshape(y_test)
            y_pred=CSI_reshape(y_pred)
            mse0 = criterion(y_pred, y_test_reshaped) 
            mse1+=mse0         
        avg_mse=mse1/num_test_batches

    return train_losses,val_losses,avg_mse




