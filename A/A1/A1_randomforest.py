import numpy as np
import lab2_landmarks as l2
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score


X,y=l2.extract_features_labels()
Y=np.array([y,-(y-1)]).T
def train_test():
    """
    Use lab2_landmark to extract dlib features of mouths and corresponding smiling labels, split into train and test sets
    :return: train and test sets
    """
    #X,y=l2.extract_features_labels()
    #Y=np.array([y,-(y-1)]).T
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
    X_train=X_train.reshape(X_train.shape[0],136)
    y_train=list(zip(*y_train))[0]
    X_test=X_test.reshape(X_test.shape[0],136)
    y_test=list(zip(*y_test))[0]
    return X_train,X_test,y_train,y_test

def cross_val(X_train,y_train):
    """
    Use cross validation to find the best number of trees
    :return: n and accuracy of validation set
    """
    n_estimator=list(range(10, 100))
    cv_scores=[]
    tr_scores=[]
    for n in n_estimator:
        rf=RandomForestClassifier(n_estimators=n)
        rf.fit(X_train, y_train)
        train_acc=rf.score(X_train, y_train)
        tr_scores.append(train_acc.mean())
        cv_score=cross_val_score(rf,X_train,y_train,cv=5)
        cv_scores.append(cv_score.mean())
    print(cv_scores)
    n_final=np.argmax(cv_scores)
    print('n_estimators=',n_final)
    #accuracy of validation
    cv_final=np.max(cv_scores)
    print('accuracy of validation set=',cv_final)
    return n_final,cv_final

X_train,X_test,y_train,y_test=train_test()
n,val_acc=cross_val(X_train,y_train)
rf=RandomForestClassifier(n_estimators=n)
rf.fit(X_train, y_train)
train_acc=rf.score(X_train,y_train)
test_acc=rf.score(X_test,y_test)
print('accuracy of train set=',train_acc)
print('accuracy of test set=',test_acc)
