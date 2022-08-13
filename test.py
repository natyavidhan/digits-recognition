from mnist import MNIST
import numpy as np
import random
import pickle

mndata = MNIST('data', gz=True)
test_in, test_out = mndata.load_testing()

clf = pickle.load(open('model.pkl', 'rb'))
result = list(clf.predict(test_in))

score = 0
for idx, val in enumerate(result):
    if val == test_out[idx]:
        score += 1

print(score / len(result)*100)