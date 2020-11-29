import numpy as np
import lab2_mouth_landmarks as l2
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


X,y=l2.extract_features_labels()
# landmarks of mouth[48,68], mouth contour[48,61], mouth[61,68]
#X=np.delete(X, slice(0, 48), axis=1)
#X=np.array(X)
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
    plt.plot(n_estimator,cv_scores,label="Validation Accuracy")
    plt.plot(n_estimator,tr_scores,label="Train Accuracy")
    plt.xlabel("Estimator Number")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim(0.8, 1)
    plt.title("Random Forest Classifier-Cross Validation")
    plt.savefig('rf.jpg')
    plt.show()
    return n_final,cv_final

X_train,X_test,y_train,y_test=train_test()
#n,val_acc=cross_val(X_train,y_train)
rf=RandomForestClassifier(n_estimators=54)
#rf=RandomForestClassifier(n_estimators=n)
rf.fit(X_train, y_train)
train_acc=rf.score(X_train,y_train)
test_acc=rf.score(X_test,y_test)
print('accuracy of train set=',train_acc)
print('accuracy of test set=',test_acc)

train_sizes=np.linspace(.1, 1.0, 5)
train_sizes,train_scores,test_scores,fit_times,_=learning_curve(rf,X_train,y_train,cv=5,train_sizes=np.linspace(.1, 1.0, 5),return_times=True)
train_scores_mean=np.mean(train_scores,axis=1)
train_scores_std=np.std(train_scores,axis=1)
test_scores_mean=np.mean(test_scores,axis=1)
test_scores_std=np.std(test_scores,axis=1)
fit_times_mean=np.mean(fit_times,axis=1)
fit_times_std=np.std(fit_times,axis=1)

#plot learning curve
plt.figure(1)
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.fill_between(train_sizes, train_scores_mean-train_scores_std,train_scores_mean+train_scores_std, alpha=0.1,color="r")
plt.fill_between(train_sizes, test_scores_mean-test_scores_std,test_scores_mean+test_scores_std, alpha=0.1,color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation Accuracy")
plt.legend(loc="best")
plt.savefig('rf_learning_curve.jpg')
plt.show()

#plot n_samples vs fit_times
plt.figure(2)
plt.plot(train_sizes, fit_times_mean, 'o-')
plt.fill_between(train_sizes, fit_times_mean-fit_times_std,fit_times_mean+fit_times_std, alpha=0.1)
plt.xlabel("Training examples")
plt.ylabel("fit_times")
plt.title("Scalability of the model")
plt.savefig('rf_sample_fittime.jpg')
plt.show()

# Plot fit_time vs score
plt.figure(3)
plt.plot(fit_times_mean, test_scores_mean, 'o-')
plt.fill_between(fit_times_mean, test_scores_mean-test_scores_std,test_scores_mean+test_scores_std, alpha=0.1)
plt.xlabel("fit_times")
plt.ylabel("Accuracy")
plt.title("Performance of the model")
plt.savefig('rf_fittime_acc.jpg')
plt.show()

