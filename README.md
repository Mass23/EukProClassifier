# EukProClassifier

EukProClassifier is a project carried out as part of the machine learning course [CS-433](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) given at EPFL during Fall 2020. It aims at improving and optimising the *EukRep* method developed by [West et al. (2018)](https://genome.cshlp.org/content/28/4/569.full), in order to retrieve eukaryotic Metagenome Assembled Genomes in  glacier-fed streams metagenome. The main goal is thus to construct an accurate and efficient (with low prediction time) binary classifier based on k-mer methods.

## How to use this code

The project consists of using a large dataset do train classifiers that can afterwards be used on the field. We use the following `scikit-learn` classifiers: `LinearSVC`, `SVC`, `LogisticRegression`, `RandomForestClassifier`, `MLPClassifier`.

### Data construction
The training dataset used is based on the NCBI genomes database (30’000+ genomes), spanning a large diversity of species. To create it, we used the program `src/GetContigs.py`.

### Data processing
Running the script `transform.py` from inside the `src` folder will compare 3 different dataset transformations with each classifier :
 - the frequency of occurrences of each k-mer for a given datapoint
 - the CLR transformation of the frequency-measuring dataset
 - a K-means transformation on the features (for 16 different number of clusters K linearly distributed in the number of features)

The raw dataset to use has to be specified in the global variable `FILENAME` at the beginning of the file. The result of the comparison is saved in a csv file in the folder `output`.

The classifiers with default parameters are used to compare the cross validated accuracy, as well as the accuracies for each class individually and the learning and prediction time. Only for the LogisticRegression are we using a big penalizing parameter `C`, so to remove the penalization (remove it by using penalty='none' takes too much time as the liblinear solver is not supported in this case). The script also compare the performances of the original *EukRep* method.

### Tuning of the methods

The script `tune_methods.py` to be run from inside the `src` folder, performs cross validation grid search over different parameters for each of the five methods. The dataset transformation to used should be specified for each method.

The raw dataset to use has to be specified in the global variable `FILENAME` at the beginning of the file. The result of the grid search is saved in a csv file in the folder `output`.

## Environment
The project has been developed with `python3.7`.
The required library for running the scripts are `numpy`, `pandas` and `sklearn`.
The library for visualization is `matplotlib`.

## Authors
The project is accomplished by team *beach_guilt* with members:
- Massimo Bourquin : [@Mass23](https://github.com/Mass23)
- Anita Dürr: [@AnitaDurr](https://github.com/AnitaDurr)
- Natasa Krco : [@Nat998](https://github.com/Nat998)
