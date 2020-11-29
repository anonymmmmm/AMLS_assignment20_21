import numpy as np
import lab2_mouth_landmarks as l2
import pickle
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve
from sklearn import svm
from sklearn import metrics

X,y=l2.extract_features_labels()

# landmarks of mouth[48,68], mouth contour[48,61], mouth[61,68]
#X=np.delete(X, slice(0, 48), axis=1)
#print(X.shape)
Y=np.array([y,-(y-1)]).T
def train_test():
    """
    Use lab2_landmark to extract dlib features of mouths and corresponding smiling labels, split into train and test sets
    :return: train and test sets
    """
    #X,y=l2.extract_features_labels()
    #Y=np.array([y,-(y-1)]).T
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
    X_train=X_train.reshape(X_train.shape[0],40)
    y_train=list(zip(*y_train))[0]
    X_test=X_test.reshape(X_test.shape[0],40)
    y_test=list(zip(*y_test))[0]
    return X_train,X_test,y_train,y_test

def cross_val(X_train,y_train):
    """
    Use cross validation to find the best penalty parameter C and appopriate kernel
    :return: C and accuracy of validation set
    """
    #c_cand=np.array([0.0001,0.001,0.01,0.1,1,10])
    #c=10 makes the program run extremely slowly
    c_cand=np.array([0.0001,0.001,0.01,0.1,1])
    #kernel=['rbf','poly','linear']
    cv_scores=[]
    #for k in range(len(kernel)):
    for i in range(len(c_cand)):
        SVM_clf=svm.SVC(C=c_cand[i],kernel='linear',gamma='auto')
        cv_score=cross_val_score(SVM_clf,X_train,y_train,cv=5)
        cv_scores.append(cv_score.mean())
    print(cv_scores)
    i=np.argmax(cv_scores)
    #cv_scores=np.array(cv_scores)
    #cv_scores=np.reshape(cv_scores,(3,6))
    #k,i=np.where(cv_scores==np.max(cv_scores))
    #choice of kernel
    #k_final=kernel[k]

    #penalty parameter C
    c_final=c_cand[i]
    #print('kernel=',k_final)
    print('C=',c_final)
    #accuracy of validation
    cv_final=cv_scores[i]
    print('accuracy of validation set=',cv_final)
    return c_final,cv_final

X_train,X_test,y_train,y_test=train_test()
c,val_acc=cross_val(X_train,y_train)
SVM_clf=svm.SVC(C=c,kernel='linear',gamma='auto')
SVM_clf.fit(X_train,y_train)
train_acc=SVM_clf.score(X_train,y_train)
test_acc=SVM_clf.score(X_test,y_test)
print('accuracy of train set=',train_acc)
print('accuracy of test set=',test_acc)