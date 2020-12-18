# AMLS_assignment20_21

Assignment for AMLS.

There are four classification problems involved. A1 and A2 are binary classification problems. B1 and B2 are multiclass classification problems. They are as the following,

A1: Gender Detection
A2: Emotion Detection
A3: Face Shape Recognition
A4: Eye Color Recognition

In A1 folder, A1.py is the logistic regression model selected finally. A1_random_forest is the random forest model used for classification, A1_SVM is the SVM model used for classification. ROC_A1.py is used for plotting ROC curves and calculating the accuracy, recall, precision and F1 score of logistic regressio, and comparing them.
In A2 folder, A2.py is the SVM model seleted finally. A2_random_forest is the random forest model used for classification, A2_logistic_regression is the LR model used for classification. ROC_A2.py is used for plotting ROC curves and calculating the accuracy, recall, precision and F1 score, and comparing them.
In B1 folder, B1.py is the CNN model selected finally.
In B2 folder, B2.py is the CNN model selected finally.

Libraries Needed:
Cv2, numpy, skimage, sklearn, matplotlib, pandas, tensorflow, keras
Also remember to download ‘haarcascade_frontalface_alt.xml’ and ‘haarcascade_mcs_mouth.xml’ in this file.
