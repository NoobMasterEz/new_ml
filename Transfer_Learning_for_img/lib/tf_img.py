#ratchanonth 1/13/2020
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard 
from tensorflow.keras.utils import get_file 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pathlib
import numpy as np
import glob


class TransferLearning(object):
    """ 
        ==========
        Parameters
        ==========
        It is a machine learning method where a model is trained on a task that can be trained (or tuned) for another task, it is very popular nowadays especially in computer vision and natural language processing problems. Transfer learning is very handy given the enormous resources required to train deep learning models. Here are the most important benefits of transfer learning:
        Speeds up training time.
        It requires less data.
        Use the state-of-the-art models that are developed by deep learning experts.
        For these reasons, it is better to use transfer learning for image classification problems instead of creating your model and training from scratch, models such as ResNet, InceptionV3, Xception, and MobileNet are trained on a massive dataset called ImageNet which contains of more than 14 million images that classifies 1000 different objects.


        :BATCH_SIZE : The batch size defines the number of samples that will be propagated through the network.
    
    """
    
    BATCH_SIZE=None
    NUM_CLASSES=None
    EPOCHS=10
    IMAGE_SHAPE=(224, 224, 3)
    TEST_DATA=None
    TRAIN_DATA=None

    def __init__(self,**kage):
        
        self.BATCH_SIZE=kage.get("BATCH_SIZE")
        self.NUM_CLASSES=kage.get("NUM_CLASSES")
        self.EPOCHS=kage.get("EPOCHS")
        self.IMAGE_SHAPE=kage.get("IMAGE_SHAPE")

  
    def load_data(self,url='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',frame="flower_photos"):
        """
            ==========
            Parameters
            ==========
            This function downloads, extracts, loads, normalizes and one-hot encodes Flower Photos dataset 
            # dowloade data sets 
            # classifiy data sets 
            # 20% validation set 80% training set
            # make the training dataset generator
            :url : from link datasets is target 
            :frame : what do you want link. i need Flower. me set defult Flower
        """
        #download the dataset and extract it
        self.data_dir = get_file(origin=url,fname=frame, untar=True)
        
        #count how many images are there 
        data_dit= pathlib.Path(self.data_dir)
        image_count=len(list(data_dit.glob('*/*.jpg')))
        print("Number of images:",image_count)
        self.CLASS_NAMES = np.array([item.name for item in list(data_dit.glob('*')) if str(item.name) != "LICENSE.txt"])
        # 20% validation set 80% training set
        self.image_generator =ImageDataGenerator(rescale=1/255,validation_split=0.2)
        


    @property
    def generator_test_train(self):
        # make the training dataset generator
        self.TEST_DATA=self.image_generator.flow_from_directory(
            directory=str(self.data_dir),
            batch_size=self.BATCH_SIZE,
            classes=list(self.CLASS_NAMES),
            target_size=(self.IMAGE_SHAPE[0],self.IMAGE_SHAPE[1]),
            shuffle=True,
            subset="training")
        
        self.TRAIN_DATA=self.image_generator.flow_from_directory(
            directory=str(self.data_dir),
            batch_size=self.BATCH_SIZE,
            classes=list(self.CLASS_NAMES),
            target_size=(self.IMAGE_SHAPE[0],self.IMAGE_SHAPE[1]),
            shuffle=True,
            subset="validation")
        

    @property
    def fit(self):
        self.load_data()
        self.generator_test_train
        print(self.TEST_DATA)
        print(self.TRAIN_DATA)

if __name__ == "__main__": 
    tfl=TransferLearning(BATCH_SIZE=32,NUM_CLASSES=5,EPOCHS=10,IMAGE_SHAPE=(224, 224, 3))
    tfl.fit