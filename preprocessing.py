import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
from scipy.signal.filter_design import gammatone
from sklearn import preprocessing as pre


def preprocessing(input, feature):
    totalData = signal.firwin(9, [0.0625, 0.46875], window = 'hamming')
    theta = signal.firwin(9, [0.0625, 0.125], window = 'hamming')
    alpha  = signal.firwin(9, [0.125, 0.203125], window = 'hamming')
    beta = signal.firwin(9, [0.203125, 0.46875], window = 'hamming')

    # Filting
    totalDataFilt = signal.filtfilt(totalData, 1, input)
    thetaFilt = signal.filtfilt(theta, 1, totalDataFilt)
    alphaFilt = signal.filtfilt(alpha, 1, totalDataFilt)
    betaFilt = signal.filtfilt(beta, 1, totalDataFilt)

    # welch
    ftheta, psdtheta = signal.welch(thetaFilt, nperseg = 256)
    falpha, psdalpha = signal.welch(alphaFilt, nperseg = 256)
    fbeta, psdbeta = signal.welch(betaFilt, nperseg = 256)

    # add to feature list
    feature.append(max(psdtheta))
    feature.append(max(psdalpha))
    feature.append(max(psdbeta))

    return feature


def main():
    featuresPreprocessed = 0
    totalFeatures = 23 * 18 * 14

    # fetch the dataset
    filePath = 'DREAMER.mat'
    print("Loading the dataset --->")
    print("<--- Dataset loaded!")

    data = sio.loadmat(filePath)
    
    print("\n~~~~~~EEG SIGNALS~~~~~~\n\nFeature Extraction Started --->")
    
    tmp = np.zeros((23, 18, 42))
    
    for k in range(0, 23):
        for j in range(0, 18):
            for i in range(0, 14):
                BSL, STM = [], []
                baseline = data['DREAMER'][0, 0]['Data'][0, k]['EEG'][0, 0]['baseline'][0, 0][j, 0][:, i]
                stimuli = data['DREAMER'][0, 0]['Data'][0, k]['EEG'][0, 0]['stimuli'][0, 0][j, 0][:, i]
                
                # preprocess the baseline and stimuli signals
                BSL = preprocessing(baseline, BSL)
                STM = preprocessing(stimuli, STM)

                segmented = np.divide(STM, BSL)
                tmp[k, j, 3 * i]     = segmented[0]
                tmp[k, j, 3 * i + 1] = segmented[1]
                tmp[k, j, 3 * i + 2] = segmented[2]

                # progress bar
                featuresPreprocessed += 1
                print("\r%d%% Features Extracted" %(featuresPreprocessed * 100 / totalFeatures), end = "")
    col = []
    for i in range(0, 14):
        col.append('psdtheta_'+str(i + 1)+'_un')
        col.append('psdalpha_'+str(i + 1)+'_un')
        col.append('psdbeta_'+str(i + 1)+'_un')
    EEG = pd.DataFrame(tmp.reshape((23 * 18, tmp.shape[2])), columns = col)
    scaler = pre.StandardScaler()
    for i in range(len(col)):
        EEG[col[i][:-3]] = scaler.fit_transform(EEG[[col[i]]])
    EEG.drop(col, axis = 1, inplace = True)
    print(EEG)
    EEG.to_csv('EEG.csv')


if __name__  ==  '__main__':
    main()