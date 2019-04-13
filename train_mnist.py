from load_data import load_mnist_data
from LeNet5 import LeNet

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

# Hyperparameter
learning_rate = 1e-3
batch_size = 100
epoches = 10

DOWNLOAD_MNIST = True # if you have the mnist dataset,you can modify it as Fasle

# define the use of GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the mnist dataset
data_train, data_test, data_loader_train, data_loader_test = load_mnist_data(batch_size, DOWNLOAD_MNIST)


# define lenet
lenet = LeNet().to(device)

# define the loss function:Cross entropy loss function which is usually used for multi-classification problems
criterian = nn.CrossEntropyLoss(reduction='sum')

# optimizer:SGD+momentum(of course,you can only choose the SGD)
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.8) # optimizer

# the process of training
for epoch in range(epoches):
    running_loss = 0.
    running_accuracy = 0.
    # input the data of every batch_size
    for data in data_loader_train:
        input, label = data
        input, label = input.to(device), label.to(device)
        # transform tensor to variable
        input, label = Variable(input), Variable(label)

        optimizer.zero_grad() # Clear the gradient before the gradient to prevent gradient accumulation

        # forward+backward
        output = lenet(input)
        loss = criterian(output,label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # the size of output:torch.size([100,10])
        # torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
        value, index = torch.max(output.data, 1)
        correct_num = (index == label).sum()
        running_accuracy += correct_num

    running_loss /= len(data_train)
    running_accuracy /= len(data_train)

    # for each epoch, print the value of loss and accuracy
    print("epoch: %d    loss: %.5f    accuracy: %.5f" % (epoch+1, running_loss, 100*running_accuracy))

# save the model
torch.save(lenet, 'lenet.pkl')






