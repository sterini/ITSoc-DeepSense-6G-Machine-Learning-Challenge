import torch
import torchvision.transforms as T
from torch.optim.lr_scheduler import _LRScheduler
import math

import plotly
import plotly.graph_objects as go

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

def str2bool(x):
    x = x.lower()
    if x == 'true':
        return True
    if x == 'false':
        return False
    raise TypeError('Wrong input!')

# Get user input
def get_user_input():
    print('\nPlease enter information below:')

    # # Ask for file name
    # file_name = input("Enter the name of the .html file that sender will send (plot.html): ")
    # print()

    # Ask for sender email
    sender_email = input("Enter sender email address (sender@gmail.com): ")
    print()
    # Ask for receiver email
    receiver_email = input("Enter the recipient's email address (receiver@gmail.com):")
    print()
    # Ask for sender password
    sender_password = input("Enter sender email password (secrete): ")
    print('\n')
    return sender_email, receiver_email, sender_password

def gen_text(args, avg_mse):
    data = ""
    if args.CAMERAS:
        data += "(Images)"
    if args.RADAR:
        data += "(Radar)"
    if args.GPS:
        data += "(GPS)"
    text = f'The email is dispatched upon the successful completion of training a model for {args.TASK}!\n'
    text += 'The email includes an attached document showcasing the performance of the model, specifically highlighting the training and validation losses.\n'
    text += f'The model was trained on {data}, and it achieved an average Mean Squared Error (MSE) loss of {avg_mse}.\n'
    text += 'Kindly download and open the attached HTML file to review the performance of the model.'
    return text


def gen_html(fig, f):
    plotly.io.write_html(fig, f'Results/{f}')

# Function that sends and email
def send_email(text, sender, receiver, sender_password, f):
    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = receiver
    message["Subject"] = "Model Pefromance Plot"
    
    attachment = MIMEApplication(open(f'Results/{f}', "rb").read(), _subtype="html")
    attachment.add_header("Content-Disposition", f"attachment; filename={f}")
    
    message.attach(attachment)
    
    text_message = text
    message.attach(MIMEText(text_message, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(sender, sender_password)

        # Send the email
        server.sendmail(sender, receiver, message.as_string())
    print("Email sent successfully.")

# Function that call send_email function, it also creates an html file that saves to your current repository
def send2email(args, fig, avg_mse, sender, receiver, sender_password, f):
    print('Generated .html file was saved to your current repository!')
    gen_html(fig, f)
    text = gen_text(args, avg_mse)
    send_email(text, sender, receiver, sender_password, f)

# Function that generates plot, displays train and validation losses
def gen_plot(args, train_losses, val_losses):
    # Plotting
    train_loss = go.Scatter(y=train_losses, name = 'train loss')
    val_loss = go.Scatter(y=val_losses, name = 'validation loss')

    # Create a Plotly figure
    fig = go.Figure(data=[train_loss, val_loss])

    title = 'Model Performance on Data of'
    if args.CAMERAS:
        title += " (Images)"
    if args.RADAR:
        title += " (Radar)"
    if args.GPS:
        title += " (GPS)"

    fig.update_layout(
        title=title,
        xaxis_title='Epochs',
        yaxis_title='Loss'
    )
    return fig

# Function to calculate model parameters
def cal_model_parameters(model):
    total_param  = []
    for p1 in model.parameters():
        total_param.append(int(p1.numel()))
    return sum(total_param)

# Preprocessing images
def normalize_image(image):
    # Convert image to float tensor
    image = image.float()
    # Normalize the image
    image /= 255.0
    trans=T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) # ImageNet mean values # ImageNet standard deviation values
    image=trans(image)
    return image

# Function to preprocess CSI data (Complex numbers) to multi channel floats
def CSI_reshape(y, csi_std=1.693e-06, target_std=0.07):
   ry = torch.real(y)
   iy= torch.imag(y)
   oy=torch.cat([ry,iy],dim=1)
   #scaling
   oy=(oy/csi_std)*target_std+0.5
   return oy

# Function to retrieve CSI data (Complex numbers) from multi channel floats
def CSI_back2original(y, csi_std=1.693e-06, target_std=0.07):
    y=((y-0.5)*csi_std)/target_std
    ry=y[:,0,:,:]
    iy=y[:,1,:,:]
    original=torch.complex(ry,iy)
    return original .reshape(-1,1,64,64)  

def decimal_to_binary(decimal_matrix, device):
    batch_size,_,l1 = decimal_matrix.shape
    binary_matrix = torch.zeros((batch_size, l1, 64), dtype=torch.int).to(device)
    for i in range(batch_size):
        for j in range(l1):
            decimal = decimal_matrix[i,0,j]
            binary = []
            while decimal > 0 and len(binary) < 64:
                decimal *= 2
                if decimal >= 1:
                    binary.append(1)
                    decimal -= 1
                else:
                    binary.append(0)

            while len(binary) < 64:
                binary.append(0)
            binary_matrix[i, j, :] = torch.tensor(binary, dtype=torch.int)
    return binary_matrix

def binary_to_decimal(binary_matrix, device):
    batch_size, l0, nb = binary_matrix.shape
    decimal_matrix = torch.zeros((batch_size, l0), dtype=torch.float).to(device)

    for i in range(batch_size):
        for j in range(l0):
            binary = binary_matrix[i, j, :]
            decimal = 0.0
            for k in range(nb):
                decimal += binary[k] * (2 ** (-k-1))

            decimal_matrix[i, j] = decimal

    return decimal_matrix

class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]

