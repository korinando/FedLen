from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def get_data(trainset, testset):
    def processing_data(dataset):
        X = dataset.data.reshape(dataset.data.shape[0], -1) / 255.0
        y = np.array(dataset.targets)
        return X, y
    X_train, y_train = processing_data(trainset)
    X_test, y_test = processing_data(testset)
    pca = PCA(0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, y_train, X_test, y_test

