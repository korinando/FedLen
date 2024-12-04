# import codecs
# import pickle
# import numpy as np
# import torchvision
#
# from sklearn.decomposition import PCA
#
#
# def get_data():
#     # Getting the CIFAR-10 dataset
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                             download=True)
#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                            download=True)
#
#     def processing_data(dataset):
#         X = dataset.data.reshape(dataset.data.shape[0], -1) / 255.0
#         y = np.array(dataset.targets)
#         return X, y
#
#     X_train, y_train = processing_data(trainset)
#     X_test, y_test = processing_data(testset)
#     pca = PCA(0.95)
#     X_train = pca.fit_transform(X_train)
#     X_test = pca.transform(X_test)
#     return (X_train, y_train), (X_test, y_test)
import codecs
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.decomposition import PCA
import numpy as np

def get_data(batch_size=128, device='cpu'):
    """
    Loads the CIFAR-10 dataset, extracts high-level features for train and test sets,
    and returns them as NumPy arrays.

    Parameters:
    - batch_size (int): Batch size for the DataLoader.
    - device (str): 'cpu' or 'cuda' for computation.

    Returns:
    - X_train (np.ndarray): Features for training data.
    - y_train (np.ndarray): Labels for training data.
    - X_test (np.ndarray): Features for test data.
    - y_test (np.ndarray): Labels for test data.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    # Load CIFAR-10 datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Load pre-trained ResNet18 and modify it to be a feature extractor
    resnet = resnet50(weights=True)
    resnet_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the final FC layer
    resnet_feature_extractor = resnet_feature_extractor.to(device)  # Move to device
    resnet_feature_extractor.eval()  # Set to evaluation mode

    # Feature extraction function
    def extract_features(data_loader, model):
        features = []
        labels = []
        with torch.no_grad():  # No gradient calculation needed
            for inputs, targets in data_loader:
                inputs = inputs.to(device)  # Move inputs to device
                feature = model(inputs)  # Extract features
                feature = feature.view(feature.size(0), -1)  # Flatten features
                features.append(feature.cpu())  # Move to CPU and store
                labels.append(targets.cpu())  # Move to CPU and store
        features = torch.cat(features, dim=0).numpy()  # Concatenate and convert to NumPy
        labels = torch.cat(labels, dim=0).numpy()
        return features, labels

    # Extract features for training and testing data
    X_train, y_train = extract_features(trainloader, resnet_feature_extractor)
    X_test, y_test = extract_features(testloader, resnet_feature_extractor)
    pca = PCA(0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return (X_train, y_train), (X_test, y_test)


def average_params(*client_params):
    avg_dict = {}
    all_labels = set(label for params in client_params for label in params.keys())

    # Average the weights and bias for each class label
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

        # Compute the average weights and bias
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
