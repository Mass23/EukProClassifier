from sklearn import svm
import numpy as np
from helpers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVC

seed = 42
# Load data
y, X, ids = load_csv_data(data_path = 'small_train.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
gamma_range = np.logspace(-5, 5, num = 20)
c_range = np.logspace(-2, 10, num = 20)

def tune_methods(X_train, y_train, X_test, y_test):
    best_accuracy = 0
    best_classifier
    best_pred
    for kern in kernels:
        classif = SVC(kernel = kern)
        
        print(testing)
            
        if(kern == 'rbf'):
            #tune rbf params
            
        classif.fit(X_train, y_train)
        pred = classif.predict(y_test)
        accuracy = accuracy_score(y_test, pred)
        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            best_classifier = classif
            best_pred = pred
        
        return best_classifier, best_accuracy, best_pred
    
#script - find best classifier, predict labels 
best_svm, best_acc, preds = tune_methods(X_train, y_train, X_test, y_test)
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Eukaryote', 'Prokaryote']))              
                
            
