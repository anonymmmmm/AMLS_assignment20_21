import numpy as np
import cv2
import os
import pandas as pd
import tensorflow
import keras
# import pickle
import shutil
import dlib
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

class B2:
  def eye(self,images_dir,labels_filename,newdir):
    """
    crop left eyes from faces and store them into a new directory
    :param images_dir: path of images
    :param labels_filename: path of labels
    :param newdir: new directory to store left eyes
    :return: a table has image names and their corresponding labels
    """
    image_paths=images_dir
    labels_file=open(labels_filename,'r')
    if os.path.exists(newdir):
      shutil.rmtree(newdir)
    os.makedirs(newdir)
    image_paths_new=[os.path.join(image_paths,l) for l in os.listdir(image_paths)]
    for img_path in image_paths_new:
      try:
        img=cv2.imread(img_path)
        #crop left eyes (48,48)
        left_eye=img[239:287,183:231]
        file_name=img_path.split('.')[1].split('/')[-1]
        cv2.imwrite(newdir+'/'+file_name+'.png',left_eye)
      except:
        pass
    images_paths_eyes=newdir
    image_paths=[os.path.join(images_paths_eyes,l) for l in os.listdir(images_paths_eyes)]
    lines=labels_file.readlines()
    eye_color_labels={line.split()[3]:str(line.split()[1]) for line in lines[1:]}
    features_all=[]
    labels_all=[]
    for img_path in image_paths:
      file_name=img_path.split('/')[-1]
      features_all.append(file_name)
      labels_all.append(eye_color_labels[file_name])
    df_image=pd.DataFrame(features_all,columns=['file_name'])
    df_label=pd.DataFrame(labels_all,columns=['eye_color'])
    df=pd.concat([df_image,df_label],axis=1)
    return df

  def preprocess_train_val(self,df,images_paths_eyes):
    """
    preprocess images
    :param df: a table has image names and their corresponding labels
    :param images_paths_eyes: the directory storing left eyes
    :return: train generator, validation generator
    """
    train,val=train_test_split(df,random_state=10)
    self.batch_size=32
    image_generator=ImageDataGenerator(rescale=1./255,validation_split=0.2,horizontal_flip=True,vertical_flip=True)
    train_generator=image_generator.flow_from_dataframe(dataframe=train,directory=images_paths_eyes,x_col='file_name',y_col='eye_color',class_mode='categorical',target_size=(32,32),batch_size=self.batch_size,subset='training')
    val_generator=image_generator.flow_from_dataframe(dataframe=val,directory=images_paths_eyes,x_col='file_name',y_col='eye_color',class_mode='categorical',target_size=(32,32),batch_size=self.batch_size,subset='validation')
    return train_generator,val_generator

  def preprocess_test(self,df,images_paths_eyes):
    """
    preprocess images
    :param df: a table has image names and their corresponding labels
    :param images_paths_eyes: the directory storing left eyes of test
    :return: test generator
    """
    test=df
    self.test_batch_size=1
    test_image=ImageDataGenerator(rescale=1./255)
    test_generator=test_image.flow_from_dataframe(dataframe=test,directory=images_paths_eyes,x_col='file_name',y_col='eye_color',class_mode='categorical',target_size=(32, 32),batch_size=self.test_batch_size,shuffle=False)
    return test_generator

  def cnn_model(self):
    """
    build a CNN model
    :return: the CNN model
    """
    self.epochs=10
    model=Sequential()
    inputShape=(32,32,3)

    # conv-->relu-->pool
    model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # conv-->relu-->pool
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # conv-->relu-->pool
    model.add(Conv2D(96,(3,3),padding="same"))
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
    learnrate=0.001
    opt=Adam(lr=learnrate, decay=learnrate/self.epochs)
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
    model.save("B2_model.h5")
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


