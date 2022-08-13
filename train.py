from mnist import MNIST
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import pickle

def train_and_save():
    mndata = MNIST('data', gz=True)
    train_in, train_out = mndata.load_training()
    test_in, test_out = mndata.load_testing()

    clf = LogisticRegression()
    clf.fit(train_in, train_out)

    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)