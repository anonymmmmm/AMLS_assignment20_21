import numpy as np
import os
import pandas as pd
import tensorflow
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Flatten,Dropout,Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
# from kerastuner import HyperModel
# from kerastuner import RandomSearch
# from kerastuner.engine.hyperparameters import HyperParameters


class B1:
  def preprocess_train_val(self,images_dir,labels_filename):
    """
    preprocess cartoon images. read files-->dataframe-->ImageDataGenerator-->train_generator, val_generator
    :param images_dir: path of cartoon images
    :param labels_filename: path of labels
    :return: train,validation generator
    """
    image_paths=images_dir
    labels_file=open(labels_filename,'r')
    lines=labels_file.readlines()
    face_shape_labels={line.split()[3]:str(line.split()[2]) for line in lines[1:]}
    df=pd.DataFrame(list(face_shape_labels.items()))
    df.columns=['file_name','face_shape']
    train,val=train_test_split(df,random_state=10)
    self.batch_size=32
    image_generator=ImageDataGenerator(rescale=1./255,validation_split=0.2,horizontal_flip=True,vertical_flip=True)
    train_generator=image_generator.flow_from_dataframe(dataframe=train,directory=image_paths,x_col='file_name',y_col='face_shape',class_mode='categorical',target_size=(32,32),batch_size=self.batch_size,subset='training')
    val_generator=image_generator.flow_from_dataframe(dataframe=val,directory=image_paths,x_col='file_name',y_col='face_shape',class_mode='categorical',target_size=(32,32),batch_size=self.batch_size,subset='validation')
    return train_generator,val_generator


  def preprocess_test(self,images_dir,labels_filename):
    """
    preprocess test images
    :param images_dir: path of cartoon images
    :param labels_filename: file name of labels csv file
    :return: test generator
    """
    test_image_paths=images_dir
    test_labels_file=open(labels_filename,'r')
    lines=test_labels_file.readlines()
    test_face_shape_labels={line.split()[3]:str(line.split()[2]) for line in lines[1:]}
    df=pd.DataFrame(list(test_face_shape_labels.items()))
    df.columns=['file_name','face_shape']
    self.test_batch_size=1
    test_image_generator=ImageDataGenerator(rescale=1./255)
    test_generator=test_image_generator.flow_from_dataframe(dataframe=df,directory=test_image_paths,x_col='file_name',y_col='face_shape',class_mode='categorical',target_size=(32,32),batch_size=self.test_batch_size,shuffle=False)
    return test_generator


  def cnn_model(self):
    """
    build a CNN model
    :return: CNN model
    """
    self.epochs=30
    model=Sequential()
    inputShape=(32,32,3)

    # conv-->relu-->pool
    model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # (conv-->relu)*2-->pool
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # (conv-->relu)*2-->pool
    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # fully connect layer-->relu
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))

    # 1 node
    model.add(Dense(5))
    model.add(Activation("softmax"))

    model.summary()
    learnrate=0.0001
    opt=Adam(lr=learnrate,decay=learnrate/self.epochs)
    compile=model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])
    return model

  def train(self,model,train_generator,val_generator):
    """
    fit the CNN model, compute train accuracy
    :param train_generator: train generator
    :param val_generator: validation generator
    :return:
    """
    self.history=model.fit_generator(generator=train_generator,steps_per_epoch=train_generator.samples//self.batch_size,epochs=self.epochs,validation_data=val_generator,validation_steps=val_generator.samples//self.batch_size)
    train_acc=self.history.history['accuracy'][-1]
    model.save("B1_model.h5")
    return train_acc

  def validation(self):
    """
    compute validation accuracy
    :return:
    """
    val_acc=self.history.history['val_accuracy'][-1]
    return val_acc

  def test(self,model,test_generator):
    """
    test the model, compute test accuracy, precision, recall and F1 score
    :param test_generator: test generator
    :return: test accuracy
    """
    samples=len(test_generator.filenames)
    predict=model.predict_generator(test_generator,samples//self.test_batch_size,verbose=1)
    pred=np.argmax(predict,axis=1)
    true=np.array(test_generator.classes)
    test_acc=accuracy_score(true,pred)
    test_recall=recall_score(true,pred,average='macro')
    test_precision=precision_score(true,pred,average='macro')
    test_f1score=f1_score(true,pred,average='macro')
    return test_acc

  def learning_curve(self):
    """
    plot learning curve about accuracy, and loss
    :return:
    """
    # accuracy over epochs
    plt.plot(self.history.history['accuracy'])
    plt.plot(self.history.history['val_accuracy'])
    plt.title('Learning Curve Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train','test'],loc='upper left')
    plt.show()
    #loss over epochs
    plt.plot(self.history.history['loss'])
    plt.plot(self.history.history['val_loss'])
    plt.title('Learning Curve Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()