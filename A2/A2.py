import cv2
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn import svm
from sklearn.model_selection import StratifiedKFold,learning_curve
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score


class A2:
    def preprocess(self,images_dir,labels_filename):
        """
        preprocess celebrity images. read files-->detect face (use the lower half part)-->detect mouth-->crop mouth-->resize-->extract hog features
        tried pca to reduce dimensions of features, but the accuracy dropped
        :param images_dir: path of celebrity images
        :param labels_filename: path of csv file
        :return: x train, y train
        """
        image_paths=[os.path.join(images_dir,l) for l in os.listdir(images_dir)]
        labels_file=open(labels_filename,'r')
        lines=labels_file.readlines()
        smile_labels={line.split()[1]:int(line.split()[3]) for line in lines[1:]}
        if os.path.isdir(images_dir):
            features_all=[]
            labels_all=[]
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        mouth_cascade=cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
        for img_path in image_paths:
            try:
                abspath=os.path.abspath(img_path)
                # file_name=img_path.split('.')[1].split('/')[-1]
                file_name=abspath.split('/')[-1]
                img=cv2.imread(img_path)
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                face=face_cascade.detectMultiScale(gray,1.1,2)
                for (x,y,w,h) in face:
                    face_rectangle=gray[y+h//2:y+h,x:x+w] # the lower half part of the face
                mouth=mouth_cascade.detectMultiScale(face_rectangle,1.1,1)
                for (mx,my,mw,mh) in mouth:
                    mouth_rectangle=face_rectangle[my:my+mh,mx:mx+mw]
                mouth_resize=cv2.resize(mouth_rectangle,(64,64))
                feature_hog=feature.hog(mouth_resize)
                features_all.append(feature_hog)
                labels_all.append(smile_labels[file_name])
            except:
                pass
        features=np.array(features_all)
        labels=np.array(labels_all)
        # gender_labels=(np.array(labels)+1)/2
        # pca=PCA(n_components=200)
        # pca.fit(features)
        # arr=pca.fit_transform(features)
        newX=features
        newY=labels
        # newY=np.array([newY,-(newY-1)]).T
        # newY=list(zip(*newY))[0]
        return newX,newY

    def svm_model(self,x,y):
        """
        build a logistic regression model
        :param x: x train
        :param y: y train
        :return: svm model
        """
        X_train=x
        y_train=y
        clf_svm=svm.SVC(C=0.1, kernel='linear', gamma='auto')
        clf_svm.fit(X_train, y_train)
        train_acc=clf_svm.score(X_train, y_train)
        joblib.dump(clf_svm, './A2_model.pkl')
        return clf_svm


    def train(self,x,y,clf_svm):
        """
        fit logistic regression model, compute train accuracy
        :param x: x train
        :param y: y train
        :param clf_svm: svm model
        :return: train accuracy
        """
        X_train=x
        y_train=y
        train_acc=clf_svm.score(X_train,y_train)
        return train_acc

    def cross_validation(self,x,y):
        """
        StratifiedKFold cross-validation
        :param x: x train
        :param y: y train
        :return:  average of 5 validation accuracy
        """
        skfold=StratifiedKFold(n_splits=5)
        #sum_train=0
        sum_val=0
        for train,test in skfold.split(x,y):
            clf_svm_val=svm.SVC(C=0.1,kernel='linear',gamma='auto')
            X_train,X_test=x[train],x[test]
            y_train,y_test=y[train],y[test]
            clf_svm_val.fit(X_train,y_train)
            #train_acc=clf_svm_val.score(X_train,y_train)
            val_acc=clf_svm_val.score(X_test,y_test)
            #sum_train+=train_acc
            sum_val+=val_acc
        #sum_train/=5
        sum_val/=5  # average of validation score
        return sum_val

    def test(self,x,y,clf_svm):
        """
        test; compute accuracy, recall, precision, and F1 score of test set
        :param x: x test
        :param y: y test
        :param clf_svm: SVM model
        :return:  accuracy
        """
        pred=clf_svm.predict(x)
        true=y
        test_acc=accuracy_score(true,pred)
        test_recall=recall_score(true,pred)
        test_precision=precision_score(true,pred)
        test_f1score=f1_score(true,pred)
        return test_acc

    def learning_curve(self,model,X_train,y_train):
        """
        plot learning curve, n_samples vs fit_times, fit_time vs score
        :param model: model
        :param X_train: X train
        :param y_train: y train
        :return:
        """
        train_sizes=np.linspace(.1,1.0,5)
        train_sizes,train_scores,test_scores,fit_times,_=learning_curve(model,X_train,y_train,cv=5,train_sizes=np.linspace(.1,1.0,5),return_times=True)
        train_scores_mean=np.mean(train_scores,axis=1)
        train_scores_std=np.std(train_scores,axis=1)
        test_scores_mean=np.mean(test_scores,axis=1)
        test_scores_std=np.std(test_scores,axis=1)
        fit_times_mean=np.mean(fit_times,axis=1)
        fit_times_std=np.std(fit_times,axis=1)
        # plot learning curve
        plt.figure(1)
        plt.grid()
        plt.xlabel("Training Examples")
        plt.ylabel("Score")
        plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="r")
        plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="g")
        plt.plot(train_sizes,train_scores_mean,'o-',color="b",label="Training Score")
        plt.plot(train_sizes,test_scores_mean,'o-',color="g",label="Cross Validation Score")
        plt.title('Learning Curve')
        plt.legend(loc="best")
        plt.show()
        # plot n_samples vs fit_times
        plt.figure(2)
        plt.grid()
        plt.plot(train_sizes,fit_times_mean,'--')
        plt.fill_between(train_sizes,fit_times_mean-fit_times_std,fit_times_mean+fit_times_std,alpha=0.1)
        plt.xlabel("Training Examples")
        plt.ylabel("Fit Times")
        plt.title("Scalability of the model")
        plt.show()
        # Plot fit_time vs score
        plt.figure(3)
        plt.grid()
        plt.plot(fit_times_mean,test_scores_mean,'o-')
        plt.fill_between(fit_times_mean,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1)
        plt.xlabel("Fit Times")
        plt.ylabel("Score")
        plt.title("Performance of the model")
        plt.show()