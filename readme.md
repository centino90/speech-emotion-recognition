## Prerequisite

### Download and Install Executables
xampp 3.3.0 
https://sourceforge.net/projects/xampp/files/XAMPP%20Windows/8.1.12/xampp-windows-x64-8.1.12-0-VS16-installer.exe

python 3.7.0
https://www.python.org/ftp/python/3.7.0/python-3.7.0-amd64.exe

### Install python packages
To intall the packages you need to run this command:
- `pip3 install -r requirements.txt`

## Reload Dataset
To insert the features into the database you need to run this command:
- `python dataset.py`

## Train the model
To train the model you need to run this command:
- `python main.py`


## Test the model
To test the model you need to run this command:
- `python test.py`


## Details
- 800 - test data
- 200 - training data


## Installable
- mysqlclient 2.1.0
- librosa 0.9.2
- numpy 1.21.6
- fastdtw 0.3.4
- pyaudio 0.2.13
- matplotlib 3.5.3
- soundfile 0.12.1
- sklearn 0.0.post1

## TODO:
- DONE trim blank points
- DONE normalize audio level
- change m4a format to wav
- DONE confusion matrix finalization
    - TP, FP, TN, FN
    - plotter
    - precision
    - recall
    - accuracy rate
- DONE training set db insertion finalization
- DONE prep script for pip installables
- DONE prep link for executables
- ADDED voiceover module to generate audio file from mic
- DONE double check Happy, Neutral, Angry CM calculation


- try adding more features
- try this implementation https://towardsdatascience.com/time-series-classification-using-dynamic-time-warping-61dcd9e143f6