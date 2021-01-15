#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


directory_file = 'C:/Users/Eeshwar/Desktop/deep learning coursera/genderdetectionface.zip'


# In[3]:


zip_ref = zipfile.ZipFile(directory_file, 'r')
zip_ref.extractall('C:/Users/Eeshwar/Desktop/deep learning coursera')
zip_ref.close()


# In[4]:


dataset = 'C:/Users/Eeshwar/Desktop/deep learning coursera/dataset1'
train_directory = os.path.join(dataset,'train')
train_men_directory = os.path.join(train_directory,'man')
train_woman_directory = os.path.join(train_directory,'woman')
test_directory = os.path.join(dataset,'test')
test_men_directory = os.path.join(test_directory,'man')
test_men_directory = os.path.join(test_directory,'woman')
validation_directory = os.path.join(dataset,'valid')
validation_men_directory = os.path.join(validation_directory,'man')
validation_woman_directory = os.path.join(validation_directory,'woman')


# In[5]:


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (150,150,3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation = tf.nn.relu),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(512, activation = tf.nn.relu),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(2, activation = tf.nn.softmax)])


# In[6]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[7]:


model.summary()


# In[8]:


train_datagen = ImageDataGenerator(
                   rescale = 1./255,
                   rotation_range  = 40,
                   width_shift_range = 0.2,
                   height_shift_range = 0.2,
                   shear_range = 0.2,
                   zoom_range = 0.2,
                   horizontal_flip = True,
                   fill_mode = 'nearest')

                 


# In[9]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[10]:


train_generator = train_datagen.flow_from_directory(
                    train_directory,
                    target_size = (150,150),
                    batch_size = 20,
                    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
                        validation_directory,
                        target_size = (150, 150),
                        batch_size = 20,
                        class_mode='binary')


# In[11]:


history = model.fit_generator(
      train_generator,
      steps_per_epoch = 80,  
      epochs= 100,
      validation_data = validation_generator,
      validation_steps = 17,
      verbose=2)


# In[15]:


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[16]:


test_generator = test_datagen.flow_from_directory(
                        test_directory,
                        target_size = (150, 150),
                        batch_size = 20,
                        class_mode='binary')


# In[19]:


output =  model.evaluate(test_generator)


# In[21]:


import matplotlib.pyplot as plt
acc = history.history['acc']
test_acc = history.output['val_acc']
loss = history.history['loss']
test_loss = history.output['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='test accuracy')
plt.title('Training and test accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='test Loss')
plt.title('Training and test loss')
plt.legend()

plt.show()

