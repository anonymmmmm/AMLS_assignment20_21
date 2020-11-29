import numpy as np
import lab2_landmarks as l2
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve
from sklearn import svm
import matplotlib.pyplot as plt


X,y=l2.extract_features_labels()
Y=np.array([y,-(y-1)]).T
def train_test():
    """
    Use lab2_landmark to extract dlib features of images and corresponding labels, split into train and test sets
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
    #plt.plot(c_cand, training_scores, label="Training Score")
    return c_final,cv_final

X_train,X_test,y_train,y_test=train_test()
#c,val_acc=cross_val(X_train,y_train)
SVM_clf=svm.SVC(C=1,kernel='linear',gamma='auto')
#SVM_clf=svm.SVC(C=c,kernel='linear',gamma='auto')
SVM_clf.fit(X_train,y_train)
train_acc=SVM_clf.score(X_train,y_train)
test_acc=SVM_clf.score(X_test,y_test)
print('accuracy of train set=',train_acc)
print('accuracy of test set=',test_acc)

train_sizes=np.linspace(.1, 1.0, 5)
train_sizes,train_scores,test_scores,fit_times,_=learning_curve(SVM_clf,X_train,y_train,cv=5,train_sizes=np.linspace(.1, 1.0, 5),return_times=True)
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
plt.savefig('svm_learning_curve.jpg')
plt.show()

#plot n_samples vs fit_times
plt.figure(2)
plt.plot(train_sizes, fit_times_mean, 'o-')
plt.fill_between(train_sizes, fit_times_mean-fit_times_std,fit_times_mean+fit_times_std, alpha=0.1)
plt.xlabel("Training examples")
plt.ylabel("fit_times")
plt.title("Scalability of the model")
plt.savefig('svm_sample_fittime.jpg')
plt.show()

# Plot fit_time vs score
plt.figure(3)
plt.plot(fit_times_mean, test_scores_mean, 'o-')
plt.fill_between(fit_times_mean, test_scores_mean-test_scores_std,test_scores_mean+test_scores_std, alpha=0.1)
plt.xlabel("fit_times")
plt.ylabel("Accuracy")
plt.title("Performance of the model")
plt.savefig('svm_fittime_acc.jpg')
plt.show()





# _, axes = plt.subplots(1, 3, figsize=(20, 5))
# axes[0].set_xlabel("Training examples")
# axes[0].set_ylabel("Score")
# #plt.grid()
# #plot learning curve
# axes[0].fill_between(train_sizes, train_scores_mean-train_scores_std,train_scores_mean+train_scores_std, alpha=0.1,color="r")
# axes[0].fill_between(train_sizes, test_scores_mean-test_scores_std,test_scores_mean+test_scores_std, alpha=0.1,color="g")
# axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
# axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
# axes[0].legend(loc="best")
# #plot n_samples vs fit_times
# axes[1].plot(train_sizes, fit_times_mean, 'o-')
# axes[1].fill_between(train_sizes, fit_times_mean-fit_times_std,fit_times_mean+fit_times_std, alpha=0.1)
# axes[1].set_xlabel("Training examples")
# axes[1].set_ylabel("fit_times")
# axes[1].set_title("Scalability of the model")
# # Plot fit_time vs score
# axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
# axes[2].fill_between(fit_times_mean, test_scores_mean-test_scores_std,test_scores_mean+test_scores_std, alpha=0.1)
# axes[2].set_xlabel("fit_times")
# axes[2].set_ylabel("Score")
# axes[2].set_title("Performance of the model")
# plt.show()

