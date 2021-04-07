"""
ResNet50 Script - Just Change paths and start script.
Library: Tensorflow
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split

import PIL
import cv2
import os
import time
from glob import glob

def readImage(path):
    """
    This function reads and resizes image from the path.
    """
    RESNET50_STANDART_IMAGE_SIZE = (224,224)
    img = np.asarray(PIL.Image.open(path).resize(RESNET50_STANDART_IMAGE_SIZE))
    return img

def showImage(image):
    plt.imshow(image)
    plt.axis("off")

TEST_PATH = "../input/animal-dataset/animal_dataset_intermediate/train/elefante_train/OIP--3aF2OpzGKcdI6FHil50qQHaFj.jpeg"
showImage(readImage(TEST_PATH))

def readDataset(root_path,blacklisted_file_names):
    """
    This function reads dataset from the root path. Root path must be like this:
    
    -=> root_path:
            class1:
                im1.jpg
                im2.jpg
                ..
            class2
            classs3 
    """
    IMAGES = []
    LABELS = []
    LABEL_MAP={}
    im_count = 0
    for index,class_ in enumerate(glob(root_path+"/*")):
        class_name = os.path.split(class_)
        
        LABEL_MAP[index] = class_name[1]
        
        for impath in glob(class_+"/*"):
            if os.path.split(impath)[1] not in blacklisted_file_names:
                img = readImage(impath)
                if img.shape != (224,224,3):
                    continue
                IMAGES.append(img)
                LABELS.append(index)
                im_count += 1

                if im_count % 500 == 0 and im_count!=0:
                    print("Reading image: {}".format(im_count))
    
    return np.array(IMAGES),np.array(LABELS),LABEL_MAP


images,labels,label_map = readDataset("../input/animal-dataset/animal_dataset_intermediate/train",["filenames.txt","filenames_elefante_train.txt"])

labels = np.asarray(pd.get_dummies(pd.Series(labels)))

labels[0]

label_map

images.shape

def getModel(CLASS_COUNT):    
    """
    This function downloads imagenet from the google depos, freeze some layers and adds blank fully 
    connected layers. It returns a ready-to-tune ResNet50 model.
    """
    base_model = ResNet50(weights="imagenet",include_top=False)
    for layer in base_model.layers[:-10]:
        layer.trainable = False

        
    model = keras.Sequential()
    model.add(layers.Input(shape=(224,224,3)))
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(CLASS_COUNT,activation="softmax"))
    
    model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
    return model

def getAugmenter():
    """
    This function returns a really simple image augmentor which just flips the images.
    If they don't help add more augmentation techniques.
    """
    aug = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                       vertical_flip=True,
                                                       validation_split=0.2
                                                      )
    
    train_gen = aug.flow(images,labels,subset="training")
    test_gen = aug.flow(images,labels,subset="validation")
    return train_gen,test_gen


train_gen,test_gen = getAugmenter()

model = getModel(5)

model.summary()

results = model.fit_generator(train_gen,validation_data=test_gen,epochs=4)

def showResults(result):
    """
    It takes a history object (the object which model.fit returns) 
    and shows the results of your model.
    """
    train_loss = result.history["loss"]
    train_accuracy = result.history["accuracy"]
    test_loss = result.history["val_loss"]
    test_accuracy = result.history["val_accuracy"]
    
    fig = plt.figure(figsize=(10,3))
    fig.add_subplot(1,2,1)
    plt.plot(train_loss,color="blue",label="Train Loss")
    plt.plot(test_loss,color="orange",label="Test Loss")
    plt.title("Loss Graph")
    plt.legend()
    
    fig.add_subplot(1,2,2)
    plt.plot(train_accuracy,color="blue",label="Train Accuracy")
    plt.plot(test_accuracy,color="orange",label="Test Accuracy")
    plt.title("Accuracy Graph")
    plt.legend()
    plt.show()
    

showResults(results)
