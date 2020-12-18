# AMLS_assignment20_21

There are four classification problems involved. A1 and A2 are binary classification problems. B1 and B2 are multiclass classification problems. 

A1: Gender Detection

A2: Emotion Detection

A3: Face Shape Recognition

A4: Eye Color Recognition

In A1 folder, A1.py is the logistic regression model selected finally. A1_random_forest.py is the random forest model used for classification, A1_SVM.py is the SVM model used for classification. ROC_A1.py is used for plotting ROC curves and calculating the accuracy, recall, precision and F1 score of logistic regression, and comparing them.

In A2 folder, A2.py is the SVM model seleted finally. A2_random_forest.py is the random forest model used for classification, A2_logistic_regression.py is the LR model used for classification. ROC_A2.py is used for plotting ROC curves and calculating the accuracy, recall, precision and F1 score of lr,svm and rf, and comparing them.

In B1 folder, B1.py is the CNN model selected finally.

In B2 folder, B2.py is the CNN model selected finally.

main.py can complete four tasks at once. A1, A2, B1 and B2_withoutmain.py present the best models and they can complete the four tasks individually.

Libraries Needed:

cv2, numpy, skimage, sklearn, matplotlib, pandas, tensorflow, keras, joblib

Also remember to download ‘haarcascade_frontalface_alt.xml’ and ‘haarcascade_mcs_mouth.xml’ in this file.

In case the main function runs an error (though it should be impossible), try to change the file name or delete the files produced during previous operation.
