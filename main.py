# Thomas Hübscher / 31.03.2021 /In spired by Youtube NeuralLine

#Image Recognition via convolutional neural network
#we use a keras dataset named cifar10

import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers

(train_images, train_labels),(test_images, test_labels) = datasets.cifar10.load_data() #will take some time

#normalize (0-1)
train_images = train_images/255
test_images = test_images/255

#in cifar10 we have labels as number, but we want string labels
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#visualize some pictures
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])

plt.show()

