# How-to
Link to download a dataset: https://ieee-dataport.org/competitions/deepverse-6g-machine-learning-challenge

unzip .zip folder and move `wireless` `RBG_images` `radar` folders and `dataset_train.csv` `dataset_test.csv` `dataset.csv` files to the same directory that contains python files from this repository! 

This code was tested on Python 3.10.0 and torch version 2.1.0+cu118

use `pip install requirements.txt` to install all necessary packages!

Note! You should install **PYTORCH** on your own. Visit this link: https://pytorch.org/get-started/locally/

run a code using
1. If you want to share results via email. First and foremost, learn how to generate a Gmail account password in section **Possible Errors**
2. e.g: `run_main.sh`
3. e.g: `python main.py --TASK task1 --GPS True --CAMERAS False --RADAR False --SHARE True --num_epochs 5`

For option number 2. You will train a model exclusively using GPS data and receive the performance metrics via email. Follow the steps prompted by the program, including filling in your email address and other required information.

Other available arguments:
```
# Parse input arguments
parser = argparse.ArgumentParser(description='Deepverse Challenge', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Task specific arguments
parser.add_argument('--TASK',type=str, default='task1', help='task name')
parser.add_argument('--GPS', type=bool, default=True, help='GPS')
parser.add_argument('--CAMERAS', type=bool, default=True, help='CAMERAS')
parser.add_argument('--RADAR', type=bool, default=True, help='RADAR')
parser.add_argument('--USE_PRESET', type=bool, default=False, help='USE_PRESET')

# Training arguments
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--patience', type=int, default=15, help='patience')
parser.add_argument('--reduction', type=int, default=8, help='reduction')
parser.add_argument('--expansion', type=int, default=20, help='expansion')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--accumulation_steps', type=int, default=20, help='accumulation steps')

# Arguments related to sharing via email
parser.add_argument('--SHARE', type=bool, default=False, help='Do you want to share results via email [True\False]')
```

# Possible Errors
*The email sending encountered an issue, and potential errors include*:
- Incorrect email address or password entry.
- Ensure you are using an application-specific password instead of your standard email account password.
   - Quick steps: https://ibb.co/ScCY1Kn
   - Detailed guide: https://medium.com/@manavshrivastava/how-to-send-emails-using-python-c89b802e0b05

*The error can occur during data loading! Your local machine should have sufficient amount of RAM memory (in my case the program uses 55GB of RAM, I use linux ubuntu)*
