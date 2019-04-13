from transform_data import transform_picture, image_to_tensor, black_to_white
from transform_data import image_to_tensor
import os
import numpy as np
import torch
from torch.autograd import Variable

# define your own data path
data_path = ".\picture"
# define the route to store the picture which are transformed, its size is 28*28
result_path = "./result/"

# define the use of GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
lenet = torch.load('lenet.pkl')

# transform the model to the test pattern
lenet.eval()

# transform your picture to the test pattern
# transform_picture(data_path, result_path)

for file in os.listdir(result_path):
    os.chdir(result_path)
    input = image_to_tensor(file)
    input = torch.unsqueeze(input, 0)
    input = input.to(device)
    input = Variable(input)

    output = lenet(input)

    value, index = torch.max(output.data, 1)

    # print the data of the prediction
    print(index.data.numpy()[0])


