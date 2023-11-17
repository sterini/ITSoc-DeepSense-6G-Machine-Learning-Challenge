# How-to
Link to download dataset: https://ieee-dataport.org/competitions/deepverse-6g-machine-learning-challenge

unzip .zip folder and move `wireless` `RBG_images` `radar` folders and `dataset_train.csv` `dataset_test.csv` `dataset.csv` files to the same directory that contains python files from this repository! 

This code was tested on Python 3.10.0 and `torch` version 2.1.0+cu118

use `pip install requirements.txt` to install all necessary packages!

Note! You should install **PYTORCH** on your own. Visit this link: https://pytorch.org/get-started/locally/

run a code using
1. If you want to share results via email. First and foremost, learn how to generate a gmail account password in section **Possible Errors**
2. `run_main.sh`
3. `python main.py --TASK task1 --GPS True --CAMERAS False --RADAR False --SHARE True`
   
For option number 2. You will train a model exclusively using GPS data and receive the performance metrics via email. Follow the steps prompted by the program, including filling in your email address and other required information.

# Possible Errors
*The email sending encountered an issue, and potential errors include*:
- Incorrect email address or password entry.
- Ensure you are using an application-specific password instead of your standard email account password.
   - Quick steps: https://ibb.co/ScCY1Kn
   - Detailed guide: https://medium.com/@manavshrivastava/how-to-send-emails-using-python-c89b802e0b05

*The error can occur during data loading! Your local machine should have sufficient amount of RAM memory (in my case the program uses 55GB of RAM, I use linux ubuntu)*
