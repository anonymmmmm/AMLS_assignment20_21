import numpy as np
import cv2
import os
import pandas as pd
import tensorflow
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

basedir='../Datasets/cartoon_set'
images_filename=os.path.join(basedir,'img')
labels_filename='labels.csv'

labels_file=open(os.path.join(basedir, labels_filename), 'r')
lines=labels_file.readlines()
face_shape_labels={line.split()[3]:str(line.split()[2]) for line in lines[1:]}
df=pd.DataFrame(list(face_shape_labels.items()))
df.columns=['file_name','face_shape']
print(df)
train,test=train_test_split(df,random_state=10)

epochs=30
batch_size=32
image_generator=ImageDataGenerator(rescale=1./255,validation_split=0.2,horizontal_flip=True,vertical_flip=True)
train_generator=image_generator.flow_from_dataframe(dataframe=train,directory=images_filename,x_col='file_name',y_col='face_shape',class_mode='categorical',target_size=(32,32),batch_size=batch_size,subset='training')
val_generator=image_generator.flow_from_dataframe(dataframe=test,directory=images_filename,x_col='file_name',y_col='face_shape',class_mode='categorical',target_size=(32,32),batch_size=batch_size,subset='validation')

def cnn_model():
  model=Sequential()
  inputShape=(32,32,3)

  model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(64,(3,3),padding="same"))
  model.add(Activation("relu"))
  model.add(Conv2D(64, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(128, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(Conv2D(128, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())
  model.add(Dense(64))
  model.add(Activation("relu"))

  model.add(Dense(5))
  model.add(Activation("softmax"))
  model.summary()
  lr=0.0001
  opt=Adam(lr=lr,decay=lr/epochs)
  compile=model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

  return model

model=cnn_model()
history=model.fit_generator(generator=train_generator,steps_per_epoch=train_generator.samples//batch_size,epochs=epochs,validation_data=val_generator,validation_steps=val_generator.samples//batch_size)

# accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Learning Curve Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Learning Curve Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()