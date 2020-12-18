import A2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Logistic Regression for A2
A2=A2.A2()
newX,newY=A2.preprocess('../Datasets/celeba/img','../Datasets/celeba/labels.csv')
X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.2,random_state=45)

lr=LogisticRegression(C=0.1,solver='liblinear')
lr.fit(X_train,y_train)
train_acc=lr.score(X_train,y_train)
test_acc=lr.score(X_test,y_test)

print('Logistic Regression train accuracy:',train_acc)
print('Logistic Regression validation accuracy:',test_acc)