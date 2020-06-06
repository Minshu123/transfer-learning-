#!/usr/bin/env python
# coding: utf-8

# In[30]:


from keras.applications import MobileNet
from glob import glob
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np 
import matplotlib.pyplot as plt
from keras.layers import Input ,Lambda, Dense, Flatten


# In[31]:


model=MobileNet(weights='imagenet',include_top=False,input_shape=(224,224,3))


# In[32]:


for layer in model.layers:
    layer.trainable=False


# In[34]:


folders=glob('/train/*')


# In[35]:


topmodel=model.output


# In[36]:


topmodel=Flatten()(model.output)


# In[37]:


topmodel=Dense(1024,activation='relu')(topmodel)


# In[38]:


topmodel=Dense(1000,activation='relu')(topmodel)


# In[39]:


topmodel=Dense(len(folders),activation='softmax')(topmodel)


# In[40]:


newmodel=Model(inputs=model.input,outputs=topmodel)


# In[41]:


newmodel.output


# In[42]:


newmodel.summary()


# In[43]:


newmodel.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
                )


# In[44]:


newmodel.summary()


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/train/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        '/validation/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
r = newmodel.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[ ]:


import tensorflow as tf

from keras.models import load_model

newmodel.save('facefeatures_new_model.h5')


# In[ ]:




