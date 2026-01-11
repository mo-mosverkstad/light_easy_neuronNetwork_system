# coding: utf-8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Setting to import files from parent directory
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def img_print(index, data, lable):
    img = data[index]
    label = lable[index]
    print(label)

    print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # reshape the image size to original size
    print(img.shape)  # (28, 28)
    print(img)
    img_show(img)

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print("x_train", x_train.shape) # (60000, 784) 784=28*28;
print("t_train", t_train.shape) # (60000,)
print("x_test", x_test.shape) # (10000, 784)
print("t_test", t_test.shape) # (10000,)

img_print(10, x_train, t_train)