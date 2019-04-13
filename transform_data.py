import cv2
import os
from PIL import Image
from torchvision import transforms
import numpy as np


transform1 = transforms.Compose([transforms.ToTensor()])


# transform a picture to the shape:1*28*28
def transform_picture(data_path, result_path):
    n = 0
    for file in os.listdir(data_path):
        n = n + 1
        os.chdir(data_path)
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
        cv2.namedWindow("1", 0)
        cv2.imshow("1", gray)
        cv2.waitKey(0)
        result = cv2.resize(gray, (28, 28))
        # cv2.imshow("1", result)
        cv2.imwrite(result_path + str(n) + ".png", result)


# if your background is white and the number is  black,you can use this function to transform it
def black_to_white(result_path):
    n = 0
    for file in os.listdir(result_path):
        os.chdir(result_path)
        img = cv2.imread(file)
        img_black = 255 - img
        cv2.imwrite(result_path + str(n) + ".png", img_black)
        n = n + 1


def image_to_tensor(file):
    img = Image.open(file).convert('L')
    img_tensor = transform1(img)
    return img_tensor

