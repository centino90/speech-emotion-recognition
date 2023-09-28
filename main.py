from model import train
import time
from utils import retrieveDatasetFromDb

curTime = time.time()

# get dataset
dataset = retrieveDatasetFromDb()

# train/test model
train(dataset, curTime)