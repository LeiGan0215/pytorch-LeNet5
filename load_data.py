import os
import torchvision
import torch
from torchvision import datasets, transforms

batch_size = 100
DOWNLOAD_MNIST = True

def load_mnist_data(batch_size, DOWNLOAD_MNIST):
    # if the data folder doesn't exist,make a data folder
    if not os.path.exists("data"):
        os.mkdir("data")

    # tensor and batch normalize
    transform = transforms.Compose([transforms.ToTensor()])#transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                    #transforms.Normalize(mean=[0.5], std=[0.5])])

    # download mnist datasets
    data_train = datasets.MNIST(root = "./data/", transform = transform, train = True, download = DOWNLOAD_MNIST)
    data_test = datasets.MNIST(root = "./data/", transform = transform, train = False)

    data_loader_train = torch.utils.data.DataLoader(dataset = data_train,batch_size = batch_size,shuffle = True)# load train data
    data_loader_test = torch.utils.data.DataLoader(dataset = data_test,batch_size = batch_size,shuffle = True)# load test data

    return data_train, data_test, data_loader_train, data_loader_test

# def transform_own_data(data_path):

# if __name__ == '__main__':
#     data_train, data_test, data_loader_train, data_loader_test = load_mnist_data(batch_size, DOWNLOAD_MNIST)

