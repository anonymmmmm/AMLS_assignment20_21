import A1
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Random Forest for A1
A1=A1.A1()
newX,newY=A1.preprocess('../Datasets/celeba/img','../Datasets/celeba/labels.csv')
X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.2,random_state=45)

rf=RandomForestClassifier(n_estimators=10)
rf.fit(X_train,y_train)
train_acc=rf.score(X_train,y_train)
test_acc=rf.score(X_test,y_test)

print('Random Forest accuracy:{:.4}',train_acc)
print('Random Forest validation accuracy:{:.4}',test_acc)