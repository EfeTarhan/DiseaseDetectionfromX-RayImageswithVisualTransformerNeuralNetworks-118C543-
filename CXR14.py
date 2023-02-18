#%% Importing Libraries
import time

st = time.time()
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import itertools
from tensorflow.keras.applications.vgg19 import VGG19
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
et=time.time()
elap = et-st
print("Libraries are imported! :",elap)

#%% Loading the data
st = time.time()
cxr14data = pd.read_csv(r"C:\Users\efeta\OneDrive\Desktop\Data_Entry_2017_v2020_3.csv")
labels = cxr14data.findings.to_list()
labels = np.array(labels)
mlb = MultiLabelBinarizer()
labels = labels.reshape((91324,1))
labels = mlb.fit_transform(labels)
labels = labels.astype('float32')
et = time.time()
elap = et-st
print("Labels are converted to binary form :",elap)

#%%Splitting data as test and training.
st = time.time()
image_train, image_test, label_train, label_test = train_test_split(('/auto/data2/sozturk/CXR8/images/images_001/images/' + cxr14data.images), labels,test_size=0.125)
image_train2= []
image_test2= []
# label_train1 = label_train[:len(image_train)//2]
# label_train2 = label_train[len(image_train)//2:]
et = time.time()
elap = et-st
print("Train test split has completed :",elap)

#%% Data Reformatting

# bu kısımda boyutları değiştiriyoz ve datayı normalize ediyoz ve eğitim için uygun hale getiriyoruz
count = 0
for i in range(len(image_train)):
    st = time.time()
    image_anchor = cv2.imread(image_train.values[i])
    image_anchor = np.array(image_anchor)
    image_anchor = cv2.cvtColor(image_anchor, cv2.COLOR_BGR2RGB)
    image_anchor = cv2.resize(image_anchor, (224, 224), interpolation=cv2.INTER_CUBIC)
    image_anchor = image.img_to_array(image_anchor)
    image_train2.append(image_anchor)
    et = time.time()
    elap = et-st
    print("Image",count," has read",elap)
    count += 1
image_train2 = np.array(image_train2)
print("Train images have  completely been read")
image_train2 = image_train2.reshape(-1,224,224,3)

#%% Data Augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(image_train2)
print("Images are completed with data augmentation")

#%% Model Generation
backbone_model=VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
model = Sequential()
model.add(backbone_model)
model.add(Flatten)
model.add(Dropout(0.3))
model.add(Dense(units=15,activation="softmax"))
print("Model has been created")

#%%
batch_size = 4
epochs = 50

history = model.fit_generator(datagen.flow(image_train2,label_train, batch_size=batch_size),
                              epochs = epochs, steps_per_epoch=image_train2.shape[0] // batch_size)
