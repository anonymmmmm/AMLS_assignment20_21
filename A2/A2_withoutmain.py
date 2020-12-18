import A2
from sklearn import svm
from sklearn.model_selection import train_test_split

# SVM for A2
A2=A2.A2()
newX,newY=A2.preprocess('../Datasets/celeba/img','../Datasets/celeba/labels.csv')
X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.2,random_state=45)

SVM_clf=svm.SVC(C=0.1,kernel='linear',gamma='auto')
SVM_clf.fit(X_train,y_train)
train_acc=SVM_clf.score(X_train,y_train)
test_acc=SVM_clf.score(X_test,y_test)

print('Random Forest accuracy:{:.4}',train_acc)
print('Random Forest validation accuracy:{:.4}',test_acc)