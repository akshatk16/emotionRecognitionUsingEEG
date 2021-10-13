import numpy as np
import pandas as pd
import scipy.io as sio


def main():
    path=u'DREAMER.mat'
    path_EEG="EEG.csv"
    data=sio.loadmat(path)
    data_EEG=pd.read_csv(path_EEG).drop(["Unnamed: 0"],axis=1)
    a=np.zeros((23,18,3))
    for k in range(0,23):
        for j in range(0,18):
            if data['DREAMER'][0,0]['Data'][0,k]['ScoreValence'][0,0][j,0]<4:
                a[k,j,0]=0
            else:
                a[k,j,0]=1 
            if data['DREAMER'][0,0]['Data'][0,k]['ScoreArousal'][0,0][j,0]<4:
                a[k,j,1]=0
            else:
                a[k,j,1]=1
            if data['DREAMER'][0,0]['Data'][0,k]['ScoreDominance'][0,0][j,0]<4:
                a[k,j,2]=0
            else:
                a[k,j,2]=1
    b=pd.DataFrame(a.reshape((23*18,a.shape[2])),columns=['Valence','Arousal','Dominance'])
    feature=pd.concat([data_EEG,b],axis=1)
    print(feature.head())
    feature.to_csv("Features.csv")


if __name__ =="__main__":
    main()