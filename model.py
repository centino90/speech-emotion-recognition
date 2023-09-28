from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import soundfile
import os
import librosa
import pickle
from fastdtw import dtw as _dtw
from utils import cmPlotter, precision, recall, accuracy
import time

emotions = sorted(['angry', 'happy', 'neutral', 'sad'])

# model trainer
def train(dataset, curTime):
    print('\n')
    print('loading dataset...')
    print('\n')
    # load training and testing data
    X_train, X_test, y_train, y_test = load_data(dataset)

    # number of samples in training data
    print("[+] Number of training samples:", X_train.shape[0])
    # number of samples in testing data
    print("[+] Number of testing samples:", X_test.shape[0])
    # number of features used
    print("[+] Number of features:", X_train.shape[1])
    # known features
    print("[+] Number of classes:", len(list(set(y_train))))
    # known features
    print("[+] Classes:", sorted(list(set(y_train))))

    # create model
    model = classification(X_train, y_train)

    # run prediction
    cm = prediction(model, X_test, y_test)

    # analyze confusion matrix
    analyze(cm)

    # save model
    save(model)

    print('\n')
    print('Time elapsed: {} seconds'.format(round(time.time() - curTime, 2)))

    # illustrate confusion matrix
    cmPlotter(cm, emotions)


# DTW helper function
def dtw(x, y):
    xr = _dtw(x, y)
    return xr[0]

# feature extractor
def extract_feature(file_name):
    with soundfile.SoundFile(file_name) as sound_file:
        # input_signal, sample_rate = librosa.load(file_name, sr=None)
        input_signal = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        stft = np.abs(librosa.stft(input_signal))
        result = np.array([])

        mfccs = np.mean(librosa.feature.mfcc(y=input_signal, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        # mel = np.mean(librosa.feature.melspectrogram(input_signal, sr=sample_rate).T,axis=0)
        # result = np.hstack((result, mel))
        # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        # result = np.hstack((result, chroma))
        # contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        # result = np.hstack((result, contrast))
        # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(input_signal), sr=sample_rate).T,axis=0)
        # result = np.hstack((result, tonnetz))
    return result

# load dataset
def load_data(dataset, test_size=0.8, train_size=0.2):
    print("[+] Test size: {:.2f}%".format(test_size*100))
    print("[+] Train size: {:.2f}%".format(train_size*100))    

    return [np.array(dataset['train_features']), np.array(dataset['test_features']), dataset['train_labels'], dataset['test_labels']]

# plot prediction then return confusion matrix
def prediction(model, X_test, y_test):
    print('\n')
    print('start prediction...')
    print('\n')

    # Predict labels for the test data
    model_prediction = model.predict(X_test)
   
    # show confusion matrix
    print("[+] Confusion matrix:")
    cm = confusion_matrix(y_test, model_prediction)
    print(cm)

    return cm

# classify dataset using KNN and DTW
def classification(X_train, y_train):
    print('\n')
    print('classifying...')
    print('\n')

    # Create KNN classifier with DTW distance metric
    model = KNeighborsClassifier(metric=dtw, n_neighbors=20)
    # Fit the classifier to the training data
    model.fit(X_train, y_train)
    print('[+] Classifier: ', 'K-Nearest Neighbors using DTW algorithm as metric')
    return model

# analzye confusion matrices
def analyze(cm):
    print('\n')
    print('analyzing matrices...')
    print('\n')
    # turn 2d array to numpy array
    ncm = np.array(cm)

    # Get one vs all type matrix
    # Anger
    print('1. Anger')
    atp = ncm[0, 0]
    atn = ncm[1:, 1:].sum()
    afp = ncm[0, 1:].sum()    
    afn = ncm[1:, 0].sum()
    print('[+] True Positive:', atp)
    print('[+] True Negative:', atn)
    print('[+] False Positive:', afp)
    print('[+] False Negative:', afn)
    aprec = precision(atp, afp)
    arec = recall(atp, afn)
    aacc = accuracy(atp, afp, atn, afn)
    print('\n')

    # Happy
    print('2. Happy')
    htp = ncm[1, 1]
    htn = ncm[2:, 2:].sum() + ncm[0, 2:].sum() + ncm[2:, 0].sum() + ncm[0, 0]
    hfp = ncm[1, 2:].sum() + ncm[1, 0]
    hfn = ncm[0, 1] + ncm[2:, 1].sum()
    print('[+] True Positive:', htp)
    print('[+] True Negative:', htn)
    print('[+] False Positive:', hfp)
    print('[+] False Negative:', hfn)
    hprec = precision(htp, hfp)
    hrec = recall(htp, hfn)
    hacc = accuracy(htp, hfp, htn, hfn)    
    print('\n')

    # neutral
    print('3. Neutral')
    ntp = ncm[2, 2]        
    ntn = ncm[:2, :2].sum() + ncm[:2, 3].sum() + ncm[3, :2].sum() + ncm[3, 3]
    nfp = ncm[2, 3] + ncm[2, :2].sum()
    nfn = ncm[:2, 2].sum() + ncm[3, 2]    
    print('[+] True Positive:', ntp)
    print('[+] True Negative:', ntn)
    print('[+] False Positive:', nfp)
    print('[+] False Negative:', nfn)
    nprec = precision(ntp, nfp)
    nrec = recall(ntp, nfn)
    nacc = accuracy(ntp, nfp, ntn, nfn)     
    print('\n')  

    # sad
    print('4. Sad')    
    stp = ncm[3, 3]
    stn = ncm[:3, :3].sum()    
    sfp = ncm[3, :3].sum()
    sfn = ncm[:3, 3].sum()    
    print('[+] True Positive:', stp)
    print('[+] True Negative:', stn)
    print('[+] False Positive:', sfp)
    print('[+] False Negative:', sfn)  
    sprec = precision(stp, sfp)
    srec = recall(stp, sfn)
    sacc = accuracy(stp, sfp, stn, sfn)
    print('\n')
    
    ## Average rates
    print('Average Rates')
    avgprec = round(((aprec + hprec + nprec + sprec) / 4) * 100, 2)
    avrec = round(((arec + hrec + nrec + srec) / 4) * 100, 2)
    avgacc =  round(((aacc + hacc + nacc + sacc) / 4) * 100, 2)
    print('[+] Avg Precision:', avgprec)
    print('[+] Avg Recall:', avrec)
    print('[+] Avg Accuracy:', avgacc)      
  

# save model to relative directory
def save(model):
    print('\n')
    print('saving model...')
    print('\n')

    # make result directory if doesn't exist yet
    if not os.path.isdir("result"):
        os.mkdir("result")

    # save model
    pickle.dump(model, open("result/kneighbors_classifier.model", "wb"))
    print("[+] Model path: result/kneighbors_classifier.model")