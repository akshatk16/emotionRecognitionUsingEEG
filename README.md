# Emotion Recognition using EEG

Emotion Recognition using EEG is a project by [Akshat Kumar Agarwal](https://github.com/akshatk16/), [Anubhav Jedia](https://github.com/jediacode/), [Deepak Sharma](https://github.com/deepak-sharma14) as a part of the requirements for the completion of the degree of Bachelor's of Technology in Electronics and Communication.

## Installation

- Install [python](https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe) and add to PATH.
- Clone this repository to the target directory by opening terminal there and typing:

```bash
git clone https://github.com/akshatk16/emotionRecognitionUsingEEG.git
```
## Usage
- Change the working directory to emotionRecognitionUsingEEG.
- Create a virutal environment using:
```bash
python -m pip install virtualenv
virtualenv env

# for windows
./env/Scripts/activate

# for Linux/MAC/Bash:
source ./env/Scripts/activate
```
- Install the required modules from the requirements.txt file by typing:
```bash
pip install -r requirements.txt
```
- Copy the DREAMER.mat file from [Zenodo](https://zenodo.org/record/546113) to the working directory.
- Run the following files one by one by typing:
```bash
python preprocessing.py
python featureExtraction.py
python main.py
```
- ***Deactivate the virtual environment.***
```bash
deactivate
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.