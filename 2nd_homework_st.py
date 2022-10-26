#!/usr/bin/env python
# coding: utf-8

# ## 2nd homework
# 
# - You should download and use the attached dataset in Moodle, it is not allowed to download this dataset from the other sources as the data configuration is different. 
# - Note that your code should be fully executable. 
# 
# 
# - You don't have to use the test dataset in the training procedure. If you need the validation data, it should be taken from the training data.  
# 
# - you can use the "fit" option which automatically takes the validation data from the training data. 
# 
# - e.g., model_1.fit(x = train_data, y =train_label, validation_split = 0.2, epochs=50)
# 
# 
# - You can either use "train_dataset" or "train_data". "train_dataset" has been formatted by Keras API with batches. 
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/DirectoryIterator
# 
# - Note that the labels are already implemented in the "train_dataset", so you can simply use "model.fit" function without adding a label, i.e., model.fit(train_dataset). 
# - Otherwise, you should also provide the data and its label as well for "train_data", i.e., model.fit(x = train_data, y = train_label)         
# - Image size and scale should be fixed as (224, 244) and [0 - 1], respectively. 
# - The given outputs in this file are the examples, you can freely choose your preferred model structures.   

# In[2]:


import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix

from tensorflow.keras.utils import to_categorical  
import tensorflow.keras.utils as util
from tensorflow.keras.layers.experimental import preprocessing

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory("C:/Users/504/Desktop/Covid19-dataset/train",
                                          target_size = (224, 224),
                                          batch_size=32,)
                                     
test_dataset = test.flow_from_directory("C:/Users/504/Desktop/Covid19-dataset/test",
                                          target_size= (224, 224),
                                          batch_size=32,)
train_dataset.class_indices
train_dataset.classes

# total training sample : 251 
# batch size = 32 
# train_dataset has 8 batches; 251 / 32 -> train_dataset[0-7][0]

train_data = []
train_label = []
test_data = []
test_label = []
for i in range(len(train_dataset)):
    train_data.append(train_dataset[i][0]) 
    train_label.append(train_dataset[i][1])    
    
train_data = np.concatenate(train_data, axis = 0)    
train_label = np.concatenate(train_label, axis = 0)

for i in range(len(test_dataset)):
    test_data.append(test_dataset[i][0]) 
    test_label.append(test_dataset[i][1])

test_data = np.concatenate(test_data, axis = 0)    
test_label = np.concatenate(test_label, axis = 0)

# or you can use entire trainig dataset (all batches are concatenated), 
# as we can use the batch option at "model.fit" function


# ### task 1 (5 points). Create 5 x 3 figures, and plot ten images per class

# In[64]:


from matplotlib import pyplot as plt
plt.figure(figsize=(20, 20))

c1 = np.where(train_label[:,0]==1)
c2 = np.where(train_label[:,1]==1)
c3 = np.where(train_label[:,2]==1)

classes = list(train_dataset.class_indices.keys())



# ### task 2 (10 points). Create your own CNN model and show its performance on test dataset with two criterias; accuracy, and confusion matrix
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

# In[81]:


def model1():
  
    
    model = keras.Model(inputs, outputs, name= "model1")
    
    return model


# In[82]:


model_1 = model1()
model_1.summary()
model_1.compile()

model_1.fit()


# In[83]:


print(model_1.evaluate())
confusion_matrix()


# ### task 3 (10 points). Transfer learning 
# - Import pre-trained model from any dataset and append a classifier to the top of the model.
# - freeze convolutional layers and train the classifier. 
# - show its performance on the test dataset

# In[4]:


transferred_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                                               include_top=False,  # True
                                               weights='imagenet',)
transferred_model.trainable = False
transferred_model.summary()


# In[5]:




model_3.compile()

model_3.fit()


# In[86]:


model_3.evaluate()
confusion_matrix()


# ### task 4 (10 points).
# - Unfreeze the convolutionla layer and find-tune the entire model to out dataset
# - Show its performance on the test dataset

# In[87]:


model_3.trainable =
model_3.fit()


# In[88]:


model_3.evaluate()
confusion_matrix()


# ### task 5 (15 points).
# - improve the model performance by applying any techniques. Even though the specific technique will not improve the performance, you should show all of your approaches here. 

# ### task 6 (15 points)
# - discuss what you have done to optize the model's performance and report your final score on test dataset. 
# - your final score will be relatively evaluated.
