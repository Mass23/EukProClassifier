from sklearn import svm
import numpy as np
from helpers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn import svm
from sklearn.svm import SVC


def plot(df):
	cs = df['C'].unique()
	gammas = df['gamma'].unique()

	fig, axs = plt.subplots(len(cs), figsize=(1+len(depths), 4*len(nb_trees)), constrained_layout=True)
	fig.suptitle('Suppport Vector Machine accuracies for different values of C and gamma parameters', fontsize=20)

	for i, c in enumerate(cs):
		axs[i].set_title('SVM with C = {}'.format(c), fontsize=16)

		glo_acc = df.loc[df['C'] == c]['accuracy'].to_numpy()
		euk_acc = df.loc[df['C'] == c]['eukaryote accuracy'].to_numpy()
		pro_acc = df.loc[df['C'] == c]['procaryote accuracy'].to_numpy()
		time = df.loc[df['C'] == c]['learning time'].to_numpy()

		axs[i].plot(gammas, glo_acc, color="red", marker='o', label='Accuracy global')
		axs[i].plot(gammas, euk_acc, color="orange", marker="o", label='Accuracy on eukaryotes')
		axs[i].plot(gammas, pro_acc, color="yellow", marker="o", label='Accuracy on prokaryotes')
		axs[i].set_xlabel("gamma",fontsize=14)
		axs[i].set_ylabel("Accuracy", fontsize=14)

		ax2 = axs[i].twinx()
		ax2.plot(gammas, time, color="blue", marker="^")
		ax2.set_ylabel("Learning time [sec]",color="blue",fontsize=14)

		axs[i].legend()

	plt.show()
	fig.savefig('SVM_C_gamma.pdf', bbox_inches='tight')

def grid_search_SVC(X, y, seed, n_jobs=None, cv=5, verbose=0):
    '''
    Performs a cross validation grid search of SVC for different values of
    parameters C and gamma. It computes the global accuracy, as well as the
    accuracy of each class. The learning time of each method is also stored.

    :return: panda DataFrame containing the cross-validation accuracy and the mean time used to learn
    '''
    # define the ranges
    c_range = np.logspace(-2, 10, num = 5) ####check ranges
    gamma_range = np.logspace(-5, 10, num = 5)
    param_grid = {'C': c_range, 'gamma':gamma_range}

    # define the scoring functions
    scorings = {'accuracy': make_scorer(accuracy_score),
            'eukaryote_accuracy':make_scorer(euk_accuracy),
            'procaryote_accuracy':make_scorer(pro_accuracy)}

    # grid search
    svc = svm.SVC(random_state=seed)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=cv,
                                scoring=scorings, refit='accuracy', verbose=verbose)
    grid_search.fit(X, y)

    # store the results in a dataframe
    df = pd.DataFrame(columns=['C', 'gamma', 'accuracy',
                    'procaryote accuracy', 'eukaryote accuracy', 'learning time'])
    for i, trial in enumerate(grid_search.cv_results_['params']):
        trial = grid_search.cv_results_['params'][i]
        trial['learning time'] = grid_search.cv_results_['mean_fit_time'][i]
        trial['accuracy'] = grid_search.cv_results_['mean_test_accuracy'][i]
        trial['procaryote accuracy'] = grid_search.cv_results_['mean_test_procaryote_accuracy'][i]
        trial['eukaryote accuracy'] = grid_search.cv_results_['mean_test_eukaryote_accuracy'][i]

        df = df.append(trial, ignore_index=True)

    df['C'] = df['C'].astype(int)
    df['gamma'] = df['gamma'].astype(int)

    plot(df)
    return df
