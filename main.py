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

#reduce images for time saving if needed
'''
train_images = train_images[:20000]
train_labels = train_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]
'''

#create model
model = models.Sequential()

#32 neurons and (3*3) as convoution matrix filter
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) # 32*32 pixel and 3 color channels
# max pooling after convolutional layer( 2,2 filter     filters for features
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
#make input 1 dimensional
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu',))
#output 10 because of 10 classes
model.add(layers.Dense(10,activation='softmax',)) # get percentage 0-1

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images,train_labels,epochs = 10, validation_data = (test_images, test_labels))

loss, accuracy = model.evaluate(test_images, test_labels)
print('loss: ', loss, ' accuracy: ', accuracy)

model.save('image_classifier.model')

#models.load_model('image_classifier.model')

