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
gamma_range = np.logspace(-5, 5, num = 5)
c_range = np.logspace(-2, 10, num = 5)

def fit_classifier(classifier, X_train, y_train, X_test, y_test):
    classif.fit(X_train, y_train)
    pred = classif.predict(y_test)
    accuracy = accuracy_score(y_test, pred)
    return accuracy, pred

def find_best_rbf(X_train, y_train, X_test, y_test):
    best_accuracy = 0
    for c in c_range:
        for g in g_range:
            classif = SVC(C = c, gamma = g)
            accuracy, pred = fit_classifier(classif, X_train, y_train, X_test, y_test)
            if(accuracy > best_accuracy):
                best_accuracy = accuracy
                best_classifier = classif
                best_pred = pred
    return best_classifier, best_accuracy, best_pred
    

def tune_methods(X_train, y_train, X_test, y_test):
    best_accuracy = 0
    for kern in kernels:
        
        print("testing svm with kernel ", kern)
            
        if(kern == 'rbf'):
            #tune rbf params with grid search
            classif, accuracy, pred = find_best_rbf(X_train, y_train, X_test, y_test)
        else:
            classif = SVC(kernel = kern)
            accuracy, pred = fit_classif(classif, X_train, y_train, X_test, y_test)
            
        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            best_classifier = classif
            best_pred = pred
        
        return best_classifier, best_accuracy, best_pred
    
#script - find best classifier, predict labels 
best_svm, best_acc, preds = tune_methods(X_train, y_train, X_test, y_test)
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Eukaryote', 'Prokaryote']))              
                
            
