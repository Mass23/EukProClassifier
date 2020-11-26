from sklearn import svm
import numpy as np
from helpers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn import svm
from sklearn.svm import LinearSVC


def plot(df):
    cs = df['C'].unique()

    fig,ax = plt.subplots()
    ax.set_title('Linear SVC', fontsize=16)

    ax.plot(df['C'], df['mean test accuracy'], color="yellow", marker="o", label = 'Global Accuracy')
    ax.plot(df['C'], df['mean_test_prokaryote_accuracy'],  color="red", marker="o", label = 'Prokaryote Accuracy')
    ax.plot(df['C'], df['mean_test_eukaryote_accuracy'],  color="orange", marker="o", label = 'Eukaryote Accuracy')
    ax.set_xlabel("C",fontsize=14)
    ax.set_ylabel("Accuracy",fontsize=14)
    ax.legend()


    ax2=ax.twinx()
    ax2.plot(df['C'], df['learning time'],color="blue",marker="o")
    ax2.set_ylabel("Learning time [sec]",color="blue",fontsize=14)
    fig.savefig('LinearSVC_C.pdf',bbox_inches='tight')
    plt.show()


def grid_search_LinearSVC(X, y, seed, n_jobs=None, cv=5, verbose=None):
	'''
	Performs a cross validation grid search of Linear SVC
	for different values of parameter C. It computes
	the global accuracy, as well as the accuracy of each class. The learning time
	of each method is also stored.

	:return: panda DataFrame containing the cross-validation accuracy and the mean time used to learn
	'''
	# define the grid
    c_range = np.logspace(-2, 10, num = 10)
    param_grid = {'C': c_range}

    # define the scoring functions
    scorings = {'accuracy': make_scorer(balanced_accuracy_score),
            'eukaryote_accuracy':make_scorer(euk_accuracy),
            'prokaryote_accuracy':make_scorer(pro_accuracy)}

    # perform the grid search
    svc = LinearSVC(random_state=seed)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=cv,
                               scoring=scorings, refit='accuracy', verbose=verbose)
    grid_search.fit(X, y)

    # store the result in a dataframe
    df = pd.DataFrame(columns=['hidden_layer_size', 'max_depth', 'accuracy',
                    'prokaryote accuracy', 'eukaryote accuracy', 'learning time'])
    for i, trial in enumerate(grid_search.cv_results_['params']):
        trial = grid_search.cv_results_['params'][i]
        trial['learning time'] = grid_search.cv_results_['mean_fit_time'][i]
        trial['accuracy'] = grid_search.cv_results_['mean_test_accuracy'][i]
        trial['prokaryote accuracy'] = grid_search.cv_results_['mean_test_prokaryote_accuracy'][i]
        trial['eukaryote accuracy'] = grid_search.cv_results_['mean_test_eukaryote_accuracy'][i]

        df = df.append(trial, ignore_index=True)

    plot(df)
    return df
