import numpy as np
import os
import pandas as pd
import tensorflow
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras.optimizers import Adam

basedir='./dataset/cartoon_set'
images_filename=os.path.join(basedir,'img')
labels_filename='labels.csv'

labels_file=open(os.path.join(basedir, labels_filename), 'r')
lines=labels_file.readlines()
face_shape_labels={line.split()[3]:int(line.split()[2]) for line in lines[1:]}
df=pd.DataFrame(list(face_shape_labels.items()))
df.columns=['file_name','face_shape']
df['file_name']=df['file_name'].astype(str)
df['face_shape']=df['face_shape'].astype(str)
train,test=train_test_split(df,random_state=10)
print(train)

epochs=20
batch_size=32
#,validate_filenames=False
image_generator=ImageDataGenerator(rescale=1./255,validation_split=0.2,horizontal_flip=True,vertical_flip=True)
train_generator=image_generator.flow_from_dataframe(dataframe=train,directory=images_filename,x_col='file_name',y_col='face_shape',class_mode='categorical',target_size=(128,128),batch_size=batch_size,subset='training')
val_generator=image_generator.flow_from_dataframe(dataframe=test,directory=images_filename,x_col='file_name',y_col='face_shape',class_mode='categorical',target_size=(128,128),batch_size=batch_size,subset='validation')
print(train_generator.samples)
print(val_generator.samples)

def cnn_model():
  model=Sequential()
  inputShape=(128,128,3)

  #SmallerVGGNet
  model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(3,3)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64,(3,3),padding="same"))
  model.add(Activation("relu"))
  model.add(Conv2D(64,(3,3),padding="same"))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128,(3,3),padding="same"))
  model.add(Activation("relu"))
  model.add(Conv2D(128,(3,3),padding="same"))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(5))
  model.add(Activation("softmax"))

  return model

model=cnn_model()
model.summary()
lr=0.0001
opt=Adam(lr=lr,decay=lr/epochs)

compile=model.compile(optimizer=opt,loss="categorical_crossentropy",metrics = ["accuracy"])

history=model.fit_generator(generator=train_generator,steps_per_epoch=train_generator.samples//batch_size,epochs=epochs,validation_data=val_generator,validation_steps=val_generator.samples//batch_size)
model.save("cnn_model_2.h5")