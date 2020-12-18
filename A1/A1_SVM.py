import A1
from sklearn import svm
from sklearn.model_selection import train_test_split

# Logistic Regression for A1
A2=A1.A1()
newX,newY=A2.preprocess('../Datasets/celeba/img','../Datasets/celeba/labels.csv')
X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.2,random_state=45)

SVM_clf=svm.SVC(C=0.1,kernel='linear',gamma='auto',probability=True)
SVM_clf.fit(X_train,y_train)
train_acc=SVM_clf.score(X_train,y_train)
test_acc=SVM_clf.score(X_test,y_test)

print('SVM train accuracy:',train_acc)
print('SVM validation accuracy:',test_acc)