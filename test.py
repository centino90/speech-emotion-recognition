import pickle
from model import extract_feature
from aud import record_to_file
from sklearn.metrics import accuracy_score
from model import emotions
import glob

# Explain class
class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "model"
        return super().find_class(module, name)

with open('result/kneighbors_classifier.model', 'rb') as f:
    # load the saved model (after training)
    unpickler = MyCustomUnpickler(f)
    model = unpickler.load()

    # TEST live voice
    # print("Please talk... Reminder: If you stopped talking for 1 second after your initial speech, the recording will end")
    # filepath = "test/test.wav"
    # # record the file (start talking)
    # record_to_file(filepath)

    # # extract features and reshape it
    # features = extract_feature(filepath).reshape(1, -1)
    # # predict
    # result = model.predict(features)[0]
    # print("[+] Result: ", result)


    # TEST prepared files
    for emotion in emotions:
        features = []
        labels = []
        path = f'test/{emotion}/*.wav'
        for file in glob.glob(path):
            # extract speech features
            feature = extract_feature(file)
            features.append(feature)
            labels.append(emotion)

        result = model.predict(features)
        print('[+] Accuracy score for', emotion)
        print(accuracy_score(labels, result))
    # filepath = 'test/happy/03-01-03-01-01-01-02-H.wav'
    

  

