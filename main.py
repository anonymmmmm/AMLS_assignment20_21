import A1.A1 as A1
import A2.A2 as A2
import B1.B1 as B1
import B2.B2 as B2

# ======================================================================================================================
# Task A1
model_A1=A1.A1()                                                              # Build model object
print('A1 is processing...')
X_train_A1,y_train_A1=model_A1.preprocess('./Datasets/celeba/img','./Datasets/celeba/labels.csv')
X_test_A1,y_test_A1=model_A1.preprocess('./Datasets/celeba_test/img','./Datasets/celeba_test/labels.csv')

lr_model_A1=model_A1.lr_model(X_train_A1,y_train_A1)                          # Logistic regression model
acc_A1_train=model_A1.train(X_train_A1,y_train_A1,lr_model_A1)                # Train model based on the training set
acc_A1_test=model_A1.test(X_test_A1,y_test_A1,lr_model_A1)                    # Test model based on the test set
print('A1 is finished')
# ======================================================================================================================
# Task A2
model_A2=A2.A2()                                                              # Build model object
print('A2 is processing...')
X_train_A2,y_train_A2=model_A2.preprocess('./Datasets/celeba/img','./Datasets/celeba/labels.csv')
X_test_A2,y_test_A2=model_A2.preprocess('./Datasets/celeba_test/img','./Datasets/celeba_test/labels.csv')

svm_model_A2=model_A2.svm_model(X_train_A2,y_train_A2)                        # SVM model
acc_A2_train=model_A2.train(X_train_A2,y_train_A2,svm_model_A2)               # Train model based on the training set
acc_A2_test=model_A2.test(X_test_A2,y_test_A2,svm_model_A2)                   # Test model based on the test set
print('A2 is finished')
# ======================================================================================================================
# Task B1
model_B1=B1.B1()                                                              # Build model object.
print('B1 is processing...')
train_generator_B1,val_generator_B1=model_B1.preprocess_train_val('./Datasets/cartoon_set/img','./Datasets/cartoon_set/labels.csv')
test_generator_B1=model_B1.preprocess_test('./Datasets/cartoon_set_test/img','./Datasets/cartoon_set_test/labels.csv')

cnn_model_B1=model_B1.cnn_model()                                             # CNN Model
acc_B1_train=model_B1.train(cnn_model_B1,train_generator_B1,val_generator_B1) # Train model based on the training set
acc_B1_test=model_B1.test(cnn_model_B1,test_generator_B1)                     # Test model based on the test set
print('B1 is finished')
# ======================================================================================================================
# Task B2
model_B2=B2.B2()                                                              # Build model object.
print('B2 is processing...')
df=model_B2.eye('./Datasets/cartoon_set/img','./Datasets/cartoon_set/labels.csv','./left_eyes')
df_test=model_B2.eye('./Datasets/cartoon_set_test/img','./Datasets/cartoon_set_test/labels.csv','./left_eyes_test')
train_generator_B2,val_generator_B2=model_B2.preprocess_train_val(df,'./left_eyes')
test_generator_B2=model_B2.preprocess_test(df_test,'./left_eyes_test')

cnn_model_B2=model_B2.cnn_model()                                             # CNN Model
acc_B2_train=model_B2.train(cnn_model_B2,train_generator_B2,val_generator_B2) # Train model based on the training set
acc_B2_val=model_B2.validation()
acc_B2_test=model_B2.test(cnn_model_B2,test_generator_B2) # Test model based on the test set.
print('B2 is finished')
# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))
