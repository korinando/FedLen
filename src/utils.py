import codecs
import pickle
import numpy as np
import torchvision

from sklearn.decomposition import PCA


def get_data():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True)

    def processing_data(dataset):
        X = dataset.data.reshape(dataset.data.shape[0], -1) / 255.0
        y = np.array(dataset.targets)
        return X, y

    X_train, y_train = processing_data(trainset)
    X_test, y_test = processing_data(testset)
    pca = PCA(0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return (X_train, y_train), (X_test, y_test)


def average_params(*client_params):
    avg_dict = {}
    all_labels = set(label for params in client_params for label in params.keys())

    for class_label in all_labels:
        weights_sum = None
        bias_sum = 0
        count = 0

        for params in client_params:
            if class_label in params:
                if weights_sum is None:
                    weights_sum = np.zeros_like(params[class_label]['weights'])
                weights_sum += params[class_label]['weights']
                bias_sum += params[class_label]['bias']
                count += 1

        if count > 0:
            avg_dict[class_label] = {
                'weights': weights_sum / count,
                'bias': bias_sum / count
            }

    return avg_dict


def dataset_split(X, y, labels):
    mask = np.isin(y, labels)
    return X[mask], y[mask]


def decode(b64_str):
    return pickle.loads(codecs.decode(b64_str.encode(), "base64"))


def encode_data(data):
    return codecs.encode(pickle.dumps(data), "base64").decode()
