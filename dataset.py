from db import insertMany, prep, deleteAllDataSet
from model import emotions, extract_feature
import glob

# create datasets table if  not exist
prep()

print('\n')
print('deleting existing dataset...')
# delete all existing dataset
deleteAllDataSet()

print('\n')
print('inserting new training dataset...')
print('\n')
dataset = []
trains = 0
tests = 0
for emotion in emotions:
    # parse train wavs
    path = f'dataset/trains/{emotion}/*.wav'
    for file in glob.glob(path):
        # extract speech features
        features = extract_feature(file)
        # add to dataset
        dataset.append((emotion, features.tobytes(), 1))
        trains+=1

    # parse test wavs
    path = f'dataset/tests/{emotion}/*.wav'
    for file in glob.glob(path):
        # extract speech features
        features = extract_feature(file)
        # add to dataset
        dataset.append((emotion, features.tobytes(), 0))
        tests+=1

print('[+] test data count:', tests)
print('[+] train data count:', trains)  

# insert in bulk
insertMany(dataset)

print('[+] Total rows inserted:', len(dataset))
classes = list(set(list(zip(*dataset))[0]))
print('[+] Total classes inserted:', len(classes))
print('[+] Classes:', sorted(classes))