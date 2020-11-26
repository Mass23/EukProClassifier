from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from helpers import *


def 


################## script

max_degree = 4
n_jobs = None
seed = 42

lin_svc = svm.LinearSVC(random_state=seed)
ker_svc = svm.SVC(random_state=seed)
log_reg = LogisticRegression(max_iter=10000, random_state=seed)
rf = RandomForestClassifier(max_depth=10, random_state=seed)
nn = MLPClassifier(solver='adam', max_iter=500, random_state=seed)
methods = {'linear svc':lin_svc, 'rbf svc':svc, 'logistic regression':log_reg, 'random forest':rf, 'neural network':nn}

df = pd.DataFrame(columns=['method', 'degree', 'accuracy', 'euk_acc', 'pro_acc',
                           'learning time', 'prediction time'])

datas = {0 : X}
for i in range(max_degree):
    poly = PolynomialFeatures(i+1)
    new_X = poly.fit_transform(X)
    datas[i+1] = new_X
            
for m in methods:
    print("Testing feature expansion for ", m)
    for deg in datas:
        X_p = datas[deg]
        #split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_p, y, random_state=seed)
        t1 = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        y_pred = clf.predict(X_test)
        t3 = time.time()
        
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        euk_acc = euk_accuracy(y_test, y_pred)
        pro_acc = pro_accuracy(y_test, y_pred)
        result = {'method' : m, 'degree' : deg, 'accuracy' : bal_acc, 'euk_acc' : euk_acc, 'pro_acc' : pro_acc,
                           'learning time' : (t2-t1), 'prediction time' : (t3 - t2)}
        df = df.append(result)
        
df.to_csv('feature_expansion.csv', index=False)
    

