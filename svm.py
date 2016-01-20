import sys
import collections
import sqlite3
import scipy.spatial
import numpy
import matplotlib.pyplot
import random
import sklearn.cluster
import pandas
import sklearn.neighbors
from math import sqrt
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import sqlite3
import DataLoader
import pickle
import time
import MLutil



# Class for SVM
class SVM(object):
    def __init__(self, svm_type = 'multi', kern='rbf', g=.0073, d=3, slack=2.8, n=.5):
        '''
        Initialize an SVM classifier based on specified paramters

        Arguments (all optional):
        svm_type: one-class or multi
        kern: rbf, poly, linear
        g: gamma
        d: degree for polynomial
        slack: error tolerance
        n: for one class only
        '''
        self.svm_type = svm_type
        if svm_type == 'one':
            self.model = sklearn.svm.OneClassSVM(kernel=kern, gamma=g, nu=n, degree=d)
        else:
            if kern == 'linear':
                self.model = sklearn.svm.LinearSVC()
            else:
                self.model = sklearn.svm.SVC(kernel=kern, gamma=g, degree=d, C=slack)

    def fit_one(self, X):
        '''
        Fit the one-class SVM, based on points X

        Arguments:
        X: list of points
        '''
        self.model.fit(X)

    def fit_multi(self, X, y):
        '''
        Fit the multiclass, based on points X, labels y

        Arguments:
        X: list of points, i.e. [(1,2), (1,3), ...]
        y: list of corresponding labels, i.e. [1, -1, ...]
        '''
        self.model.fit(X, y)

    def fit(self, X, y=None):
        '''
        Generic fitting, will call fit_one or fit_multi
        '''
        if self.svm_type == 'one':
            self.fit_one(X)
        else:
            self.fit_multi(X, y)

    def predict(self, X):
        '''
        Returns predictions for new list of points, X
        '''
        return self.model.predict(X)

    def decision_function(self, X):
        '''
        Returns distance from decision boundary for new list of points,
        '''
        return self.model.decision_function(X)

    # Plot the decision boundary
    def plot_decision_boundary(self, c='purple', fig=None):
        '''
        Plotting function
        '''
        xx, yy = numpy.meshgrid(numpy.linspace(0, 200, 200), numpy.linspace(0, 200, 200))
        Z = self.model.predict(numpy.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        if fig:
            fig.contour(xx, yy, Z, colors=c)
        else:
            matplotlib.pyplot.contour(xx, yy, Z, colors=c)

    def get_support_vectors(self):
        return self.model.support_vectors_


def svm_main(dataset, pickle_model):
    data = DataLoader.load_kaggle_mnist(dataset, neural=False)
    classifier = SVM()
    start = time.time()
    print("Fitting the svm")
    X = numpy.array(data[0][0])
    X = X/255.0*2 - 1
    print(X)
    Y = numpy.array(data[0][1])
    print(len(X))
    print(len(Y))
    del data
    classifier.fit_multi(X, Y)
    fin = time.time() - start
    print("Awesome, the SVM has been fit, only took {0} seconds".format(fin))
    pickle.dump(classifier, open(pickle_model, "wb"))

def predict_main(classifier_pickle):
    data = DataLoader.load_kaggle_mnist("mnist_train.csv", neural=False)
    X = numpy.array(data[2][0])
    X = X/255.0*2 - 1
    Y = numpy.array(data[2][1])
    predictor = MLutil.Predictor(classifier_pickle, 'SVM')
    predicted_values = predictor.make_prediction(X)

    predAnalysis = MLutil.PredictionAccuracies(predicted_values, Y)
    print(predAnalysis.get_misclass_rate())
    print(predAnalysis.get_indicies_misclassifications())

    pickle.dump(predAnalysis.get_indicies_misclassifications(), open("svm_indicies.p", "wb"))
    return predAnalysis.get_indicies_misclassifications()

if __name__ == '__main__':
    #svm_main("mnist_train.csv")
    predict_main('svm_model.p')