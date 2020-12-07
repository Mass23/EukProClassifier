# EukProClassifier

EukProClassifier is a project carried out as part of the machine learning course [CS-433](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) given at EPFL during Fall 2020. It aims at improving and optimising the *EukRep* method developed by [West et al. (2017)](https://genome.cshlp.org/content/28/4/569.full), in order to retrieve eukaryotic Metagenome Assembled Genomes in  Glacier-fed streams metagenome.

## Project structure

The project consists of using a large dataset do train classifiers that can afterwards be used on the field. We use the following `scikit-learn` classifiers: `LinearSVC`, `SVC`, `LogisticRegression`, `RandomForestClassifier`, `MLPClassifier`.

In this project, we use the scikit-learn machine learning package in order to build on the previous method by West et al. (2017) that uses its LinearSVC function, This is also motivated by reproducibility purposes. Before fine-tuning the parameters of each method in section ??, we compare different data processing steps in section ??. The tuning of each method will be done using the optimized dataset found. Finally, we study the influence of the k-mer size in section ??

### Data construction
The training dataset used is based on the NCBI genomes database (30’000+ genomes), spanning a large diversity of species. To create it, we use the script `src/GetContigs.py`,

### Data processing
Running the script `transform.py` from inside the `src` folder will compare 3 different dataset transformations with each classifier:
 - the frequency of appearance of each k-mer for a given datapoint
 - the CLR transformation of the frequency-measuring dataset
 - a K-means transformation on the features (for K = 64, 128, 256 and 512 clusters)
The classifiers with default parameters are used to compare the cross validated accuracy, as well as the accuracies for each class individually and the learning and prediction time. Only for the LogisticRegression are we using a big penalizing parameter `C`, so to remove the penalization. The script also compare the performances of the original *EukRep* method.

### Tuning of the methods

The script `tune_methods.py` to be run from inside the `src` folder, performs cross validation grid search over different parameters for each of the five methods. The dataset transformation to used should be specified for each method.

## Environment
The project has been developed with `python3.7`.
The required library for running the scripts are `numpy`, `pandas` and `sklearn`.
The library for visualization is `matplotlib`.

## Authors
The project is accomplished by team *beach_guilt* with members:
- Massimo Bourquin :
- Anita Dürr: [@AnitaDurr](https://github.com/AnitaDurr)
- Natasa Krco :
