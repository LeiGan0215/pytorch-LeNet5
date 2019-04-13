from load_data import load_mnist_data
import torch
from torch.autograd import Variable


# Hyperparameter
batch_size = 100
DOWNLOAD = False

# define the use of GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
lenet = torch.load('lenet.pkl')

# transform the model to the test pattern
lenet.eval()

# define the use of GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the mnist dataset
data_train, data_test, data_loader_train, data_loader_test = load_mnist_data(batch_size, DOWNLOAD)

test_accuracy_num = 0

for data in data_loader_test:
    input, label = data
    input, label = input.to(device), label.to(device)
    # transform tensor to variable
    input, label = Variable(input), Variable(label)

    output = lenet(input)

    value, index = torch.max(output.data, 1)
    # print(index.data.numpy()[0])
    correct_num = (index == label).sum() # correct_num is a tensor
    test_accuracy_num += correct_num.item() # transform the tensor to int

test_accuracy = test_accuracy_num / len(data_test)

print("Accuracy of the network on the %d test images:%.5f%%" % (len(data_test), 100*test_accuracy))


