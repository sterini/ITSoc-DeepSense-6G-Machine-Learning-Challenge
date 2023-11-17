Download dataset
https://ieee-dataport.org/competitions/deepverse-6g-machine-learning-challenge

unzip .zip folder and move `wireless` `RBG_images` `radar` folders and `dataset_train.csv` `dataset_test.csv` `dataset.csv` files to the same directory that contains python files from this repository! 

run a code using
1. `run_main.sh`
2. `python main.py --TASK task1 --GPS True --CAMERAS False --RADAR False --SHARE True`
   
For option number 2. You will train a model exclusively using GPS data and receive the performance metrics via email. Follow the steps prompted by the program, including filling in your email address and other required information.

The email sending encountered an issue, and potential errors include:
- Incorrect email address or password entry.
- Ensure you are using an application-specific password instead of your standard email account password.

For guidance on generating application-specific passwords, refer to the following resources:
- Quick steps: https://ibb.co/ScCY1Kn
- Detailed guide: https://medium.com/@manavshrivastava/how-to-send-emails-using-python-c89b802e0b05
