import A1
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,roc_curve,recall_score,f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression

# Plot ROC curves of SVM, logistic regression and random forest
# Calculate train accuracy, validation accuracy, precision, recall and F1 score
A1=A1.A1()
newX,newY=A1.preprocess('../Datasets/celeba/img','../Datasets/celeba/labels.csv')

X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.2,random_state=45)

#SVM
SVM_clf=svm.SVC(C=0.1,kernel='linear',gamma='auto',probability=True)
SVM_clf.fit(X_train,y_train)
train_acc=SVM_clf.score(X_train,y_train)
test_acc=SVM_clf.score(X_test,y_test)
y_pred=SVM_clf.predict(X_test)
y_proba=SVM_clf.predict_proba(X_test)[::,1]
test_precision=precision_score(y_test,y_pred)
test_recall=recall_score(y_test,y_pred)
test_f1=f1_score(y_test,y_pred)

#Random Forest
rf=RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
y_pred1=rf.predict(X_test)
y_proba1=rf.predict_proba(X_test)[::,1]
train_acc1=rf.score(X_train,y_train)
test_acc1=rf.score(X_test,y_test)
test_precision1=precision_score(y_test,y_pred1)
test_recall1=recall_score(y_test,y_pred1)
test_f11=f1_score(y_test,y_pred1)

#Logistic Regression
lr=LogisticRegression(C=0.1,solver='liblinear')
lr.fit(X_train,y_train)
train_acc2=lr.score(X_train,y_train)
y_pred2=lr.predict(X_test)
y_proba2=lr.predict_proba(X_test)[::,1]
test_acc2=lr.score(X_test,y_test)
test_precision2=precision_score(y_test,y_pred2)
test_recall2=recall_score(y_test,y_pred2)
test_f12=f1_score(y_test,y_pred2)

#Plot ROC Curves
test_auc=roc_auc_score(y_test,y_proba)
test_auc1=roc_auc_score(y_test,y_proba1)
test_auc2=roc_auc_score(y_test,y_proba2)
fpr,tpr,thresholds=roc_curve(y_test,y_proba)
fpr1,tpr1,thresholds1=roc_curve(y_test,y_proba1)
fpr2,tpr2,thresholds2=roc_curve(y_test,y_proba2)

plt.figure()
plt.plot(fpr,tpr,color='red',label='SVM ROC Curve (area = %0.4f)' %test_auc)
plt.plot(fpr1,tpr1,color='darkorange',label='RF ROC Curve (area = %0.4f)' %test_auc1)
plt.plot(fpr,tpr,color='green',label='LR ROC Curve (area = %0.4f)' %test_auc2)
plt.plot([0,1],[0,1],color='navy',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('A2 Roc Curve')
plt.legend(loc="lower right")
plt.show()

print('SVM train set： accuracy:',train_acc)
print('RF train set： accuracy:',train_acc1)
print('LR train set： accuracy:',train_acc2)
print('SVM validation set： accuracy:{:.4},precision:{:.4},recall:{:.4},f1_score:{:.4}'.format(test_acc,test_precision,test_recall,test_f1))
print('RF validation set： accuracy:{:.4},precision:{:.4},recall:{:.4},f1_score:{:.4}'.format(test_acc1,test_precision1,test_recall1,test_f11))
print('LR validation set： accuracy:{:.4},precision:{:.4},recall:{:.4},f1_score:{:.4}'.format(test_acc2,test_precision2,test_recall2,test_f12))