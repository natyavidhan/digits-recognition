from mnist import MNIST
from sklearn.linear_model import LogisticRegression
import pickle

def train_and_save():
    mndata = MNIST('data', gz=True)
    train_in, train_out = mndata.load_training()

    clf = LogisticRegression()
    clf.fit(train_in, train_out)

    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)

if __name__ == '__main__':
    train_and_save()