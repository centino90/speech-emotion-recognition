from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import random
import numpy as np
from model import load_data, train
from utils import retrieveDatasetFromDb
import matplotlib.pyplot as plt

def classifyNN(k:int, labels, dataset, test) -> str:

    x_train, x_test, y_train, y_test = load_data(dataset)
   
    idxs=range(0,x_train.shape[0])
    n=x_train.shape[0]
    distances=[]
    counters={}
    c=1;
    max_value=0

    for r in range(n):
            distances.append(dtw.distance(test, x_train[idxs[r]],window=10,use_pruning=True))
    NN=sorted(range(len(distances)), key=lambda i: distances[i], reverse=False)[:k]
        
    for l in labels.values():
        counters[l]=0

    for r in NN:
        l=labels[y_train[r]]
        counters[l]+=1
        if (counters[l])>max_value:
            max_value=counters[l]
        #print('NN(%d) has label %s' % (c,l))
        c+=1
        
    # find the label(s) with the highest frequency
    keys = [k for k in counters if counters[k] == max_value]

    # in case of a tie, return one at random
    return (random.sample(keys,1)[0])

labels = {'angry':'angry', 'happy':'happy', 'neutral':'neutral',
          'sad':'sad'}
dataset = retrieveDatasetFromDb()
datasets = list(dataset)
testdata = list(datasets[0])[0]
testlabel = list(datasets[1])[0]

res = classifyNN(20, labels, dataset, testdata)

# plt.plot(testdata, label=labels[testlabel], linewidth=2)
# plt.xlabel('Samples @50Hz')
# plt.legend(loc='upper left')
# plt.tight_layout()

print(testlabel)
print(res)