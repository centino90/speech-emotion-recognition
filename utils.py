import matplotlib.pyplot as plt
import numpy as np
from db import retrieveDataSet
import itertools

# bytes to float64 converter
def float64Converter(a):
    return (np.frombuffer(a, 'float64'))

# retrieve dataset from database
def retrieveDatasetFromDb():
    print('\n')
    print('retrieving dataset from db...')
    print('\n')
    data = retrieveDataSet()
    print('[+] Rows retrieved: ', len(data))
    
    trains = list(filter(lambda d: d[3] == 1, data))
    tests = list(filter(lambda d: d[3] == 0, data))

    train_features = list(map(float64Converter, list(zip(*trains))[2]))
    train_labels = list(zip(*trains))[1]

    test_features = list(map(float64Converter, list(zip(*tests))[2]))
    test_labels = list(zip(*tests))[1]

    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels
    }

# confusion matrix plotter
def cmPlotter(cm, emotions):
    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Speech-Emotion-Recognition Multi-class Confusion Matrix')
    plt.colorbar()

    # tilt Y labels
    target_names = emotions
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    # normalize
    normalize = False
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot the confusion matrix object  
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    # show display
    plt.tight_layout()
    plt.ylabel('Expected')
    plt.xlabel('Predicted')
    plt.show()

# precision helper
def precision(tp, fp):
    prec = tp/(tp+fp)
    print('[+] Precision: {:.2f}%'.format(prec*100))
    return prec

# recall helper
def recall(tp, fn):
    recall = tp/(tp+fn)    
    print('[+] Recall: {:.2f}%'.format(recall*100))
    return recall

# accuracy rate helper
def accuracy(tp, fp, tn, fn):
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    print("[+] Accuracy: {:.2f}%".format(accuracy*100))
    return accuracy