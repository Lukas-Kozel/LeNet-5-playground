#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Preprocessor():
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
    
    def __load_and_preprocess_image(self,image_path):
        image = Image.open(image_path)
        print(image.size)
        image = image.resize((28, 28))  # Ensure the image is 28x28
        image = np.array(image, dtype=np.float32)
        image = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)
        image = image / 255.0
        image = 1.0 - image  # Invert colors
        #print(image.shape)
        return image

    def preprocess_images(self)->tuple:
        for folder in os.listdir(self.dataset_path):
            #print(folder)
            folder_path = os.path.join(self.dataset_path,folder+"/"+folder)
            if os.path.isdir(folder_path):
                #print(folder_path)
                label = int(folder)
                for image_file in os.listdir(folder_path):
                    if image_file.endswith('.png'):
                        image_path = os.path.join(folder_path,image_file)
                        image = self.__load_and_preprocess_image(image_path=image_path)
                        self.images.append(image)
                        self.labels.append(label)
        train_images = tf.convert_to_tensor(self.images,dtype=tf.float32)
        train_labels = tf.convert_to_tensor(self.labels,dtype=tf.int32)
        return (train_images,train_labels)
    
    def plot_random_preprocessed_image(self):
        plt.imshow(self.images[random.randint(0,107730)], cmap='gray')  # 'cmap' specifies that we want to see the image in grayscale
        plt.title('Random Image from Training Data test')
        plt.colorbar()
        plt.show()
        print(self.images[random.randint(0,107730)])
    #print(train_images)


