# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

print(os.path.abspath(__file__))
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
img_path = os.path.join(os.path.dirname(script_dir), 'dataset', 'lena.png')
print(img_path)
img = imread(img_path) # Load image
plt.imshow(img)

plt.show()