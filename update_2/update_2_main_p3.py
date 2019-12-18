#!/usr/local/bin/python3
# coding: utf-8

'''
Filename: update_2_main_p3.py
Authors: Jonathan A. Gibson, Andrew P. Worley, & Raunak S. Hakya
Objectives:
	Type-based model: REPO_TYPE
	Size-based model: ADD_LOC, DEL_LOC, TOT_LOC
	Time-based model: PRIOR_AGE
	Full model: ADD_LOC, DEL_LOC, TOT_LOC, PRIOR_AGE, REPO_TYPE

	Repeat the following steps for type-based model, size-based model, time-based model, and full model:
	1. Take CSV as input, separate out independent columns(s), and the dependent column is SECU_FLAG.
	2. Apply Decision Tree, Random Forest, ANN (Artificial Neural Network), kNN (k Nearest Neighbor), & Naïve Bayes classification algorithms.
	3. Apply 10-by-10 fold cross validation, and then report prediction accuracy using precision, recall, and F-measure.
Notes:
	Naïve Bayes Reference(s):
		https://scikit-learn.org/stable/modules/naive_bayes.html

	k Nearest Neighbor (kNN) Reference(s):
		https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

	Desicion Tree Reference(s):
		https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
		http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html

	Artificial Neural Network (ANN) Reference(s):
		https://scikit-learn.org/stable/modules/neural_networks_supervised.html

	Random Forest Reference(s):
		https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

	Cross Validation Reference(s):
		https://scikit-learn.org/stable/modules/cross_validation.html
		https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
		https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

	F-Score Reference(s):
		https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
		https://deepai.org/machine-learning-glossary-and-terms/f-score
		https://deepai.org/machine-learning-glossary-and-terms/precision-and-recall
		P = Precision = TP / (TP + FP)
		R = Recall = TP / (TP + FN)
		F = F1_Score = 2 * [(P * R) / (P + R)]
'''

#--------------------------------------------- Import Necessary Libraries --------------------------------------------#

from sklearn.tree import DecisionTreeClassifier # for decision tree classification
from sklearn.ensemble import RandomForestClassifier # for random forest classification
from sklearn.neural_network import MLPClassifier # artificial neural network classification
from sklearn.neighbors import KNeighborsClassifier # for k nearest neighbor classification
from sklearn.naive_bayes import GaussianNB # for naïve bayes classification
import time # for getting model runtime
import pandas as pd # for Dataframe support
import numpy as np # for mathmatical support
import os.path # for testing file existance
import sys # for managing command line arguments
import pdb # for Python debugging

#------------------------------------------- Import Custom Utility Modules -------------------------------------------#

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utilities")) # add "../utilities" to module import search path
import logger # for custom "Logger" class
import update_2_utils_p3 as u2u_p3 # for custom "update_2_utils" utility functions

#-------------------------------------------- Check Command Line Arguments -------------------------------------------#

print("\nProgram Name: ", sys.argv[0])
print("Number of command line arguments:", len(sys.argv))
print("The command line argument(s) are:" , str(sys.argv))

if(len(sys.argv) != 2):
	print("ERROR::160::BAD_ARGUMENTS")
	print("Run program with the following format: \'python update_2_main_p#.py \"<log_file>\"\'; alternatively use \'make run2\' or \'make run3\' for Python2 and Python3 appropriate scripts respectively.\n")
	sys.exit()

#------------------------------------------------- Load Dataset File -------------------------------------------------#

valid_dataset_names = ["update_2_dataset.csv", "dataset.csv", "Dataset.csv", "DATASET.csv"]

for d in valid_dataset_names:
	if(os.path.exists(d)):
		print("\nLoading dataset file...")
		df = pd.read_csv(d)
		print("Done.")
		break
	else:
		print("ERROR::002::FILE_NOT_FOUND")
		print("Place project dataset file in root directory with one of the following valid dataset file names:\n", valid_dataset_names, "\n")
		sys.exit()

print("\nDataset datatypes:")
print(df.dtypes)
print("\nDataset original shape:", df.shape)
print("\nOriginal dataset:")
print(df.head())

#------------------------------------------------------- Setup -------------------------------------------------------#

target_df, targets = u2u_p3.encode_target(df, "SECU_FLAG") # encoding dataframe with "SECU_FLAG" as classification target
targets_dict = u2u_p3.col_count_perc(df, "SECU_FLAG", len(df.index), 2) # count unique targets and calculate percentiles
enc_df, repo_types = u2u_p3.encode_string_column(target_df, "REPO_TYPE") # encoding the "REPO_TYPE" column
repo_types_dict = u2u_p3.col_count_perc(df, "REPO_TYPE", len(df.index), 3) # count repo types and calculate percentiles

print("\nEncoded Targets:")
for i in range(0, len(targets)):
	print("\t", targets[i], "–> Encoded as:", str(i)+',',\
													"Count:", int(targets_dict[targets[i]][0]), "/ 300056,",\
													"Percentage:", str(targets_dict[targets[i]][1]*100)+"%")

print("\nEncoded Repository Types: ")
for j in range(0, len(repo_types)):
	print("\t", repo_types[j], "–> Encoded as:", str(j)+',',\
														 "Count:", int(repo_types_dict[repo_types[j]][0]), "/ 300056,",\
														 "Percentage:", str(repo_types_dict[repo_types[j]][1]*100)+"%")
	
fold_count = 10 # number of folds for cross-validation
secs_in_min = 60 # number of seconds in a minute

type_model_features = list(enc_df.columns[11:12]) # REPO_TYPE
size_model_features = list(enc_df.columns[3:6]) # ADD_LOC, DEL_LOC, TOT_LOC
time_model_features = list(enc_df.columns[8:9]) # PRIOR_AGE
full_model_features = list(enc_df.columns[3:6]) + list(enc_df.columns[8:9]) + list(enc_df.columns[11:12]) # ADD_LOC, DEL_LOC, TOT_LOC, PRIOR_AGE, REPO_TYPE

print("\n********************************************************* STARTING CLASSIFICATION *********************************************************")

#-------------------------------------------- Decision Tree Classification -------------------------------------------#

print("\nUsing Decision Tree Classification:\n")
dtc = DecisionTreeClassifier(random_state=1) # define the decision tree classifier
dtc_start = time.time()

#-------------------- Type-Based Model -------------------#

u2u_p3.classification(dtc, type_model_features, enc_df, fold_count, "Type", targets)

#-------------------- Size-Based Model -------------------#

u2u_p3.classification(dtc, size_model_features, enc_df, fold_count, "Size", targets)

#-------------------- Time-Based Model -------------------#

u2u_p3.classification(dtc, time_model_features, enc_df, fold_count, "Time", targets)

#----------------------- Full Model ----------------------#

u2u_p3.classification(dtc, full_model_features, enc_df, fold_count, "Full", targets)

#---------------------- DTC Run Time ---------------------#

dtc_end = time.time()
dtc_run = (dtc_end - dtc_start)
dtc_mins, dtc_secs = divmod(dtc_run, secs_in_min)
print("\tThe decision tree classifier produced results for all 4 models in %0.2f minute(s) and %0.2f seconds." % (dtc_mins, dtc_secs), "\n")

#-------------------------------------------- Random Forest Classification -------------------------------------------#

print("Using Random Forest Classification (with 50 trees):\n")
rfc = RandomForestClassifier(n_estimators=50, random_state=2) # define the random forest classifer
rfc_start = time.time()

#-------------------- Type-Based Model -------------------#

u2u_p3.classification(rfc, type_model_features, enc_df, fold_count, "Type", targets)

#-------------------- Size-Based Model -------------------#

u2u_p3.classification(rfc, size_model_features, enc_df, fold_count, "Size", targets)

#-------------------- Time-Based Model -------------------#

u2u_p3.classification(rfc, time_model_features, enc_df, fold_count, "Time", targets)

#----------------------- Full Model ----------------------#

u2u_p3.classification(rfc, full_model_features, enc_df, fold_count, "Full", targets)

#---------------------- RFC Run Time ---------------------#

rfc_end = time.time()
rfc_run = (rfc_end - rfc_start)
rfc_mins, rfc_secs = divmod(rfc_run, secs_in_min)
print("\tThe random forest classifier produced results for all 4 models in %0.2f minute(s) and %0.2f seconds." % (rfc_mins, rfc_secs), "\n")

#------------------------------------------------- ANN Classification ------------------------------------------------#

print("Using Artificial Neural Network Classification:\n")
ann = MLPClassifier(random_state=3)
ann_start = time.time()

#-------------------- Type-Based Model -------------------#

u2u_p3.classification(ann, type_model_features, enc_df, fold_count, "Type", targets)

#-------------------- Size-Based Model -------------------#

u2u_p3.classification(ann, size_model_features, enc_df, fold_count, "Size", targets)

#-------------------- Time-Based Model -------------------#

u2u_p3.classification(ann, time_model_features, enc_df, fold_count, "Time", targets)

#----------------------- Full Model ----------------------#

u2u_p3.classification(ann, full_model_features, enc_df, fold_count, "Full", targets)

#---------------------- ANN Run Time ---------------------#

ann_end = time.time()
ann_run = (ann_end - ann_start)
ann_mins, ann_secs = divmod(ann_run, secs_in_min)
print("\tThe artificial neural network classifier produced results for all 4 models in %0.2f minute(s) and %0.2f seconds." % (ann_mins, ann_secs), "\n")

#------------------------------------------------- kNN Classification ------------------------------------------------#

print("Using k Nearest Neighbor Classification (with 3 neighbors threshold):\n")
knn = KNeighborsClassifier(n_neighbors=3) # define the k nearest neighbor classifier
knn_start = time.time()

#-------------------- Type-Based Model -------------------#

u2u_p3.classification(knn, type_model_features, enc_df, fold_count, "Type", targets)

#-------------------- Size-Based Model -------------------#

u2u_p3.classification(knn, size_model_features, enc_df, fold_count, "Size", targets)

#-------------------- Time-Based Model -------------------#

u2u_p3.classification(knn, time_model_features, enc_df, fold_count, "Time", targets)

#----------------------- Full Model ----------------------#

u2u_p3.classification(knn, full_model_features, enc_df, fold_count, "Full", targets)

#---------------------- kNN Run Time ---------------------#

knn_end = time.time()
knn_run = (knn_end - knn_start)
knn_mins, knn_secs = divmod(knn_run, secs_in_min)
print("\tThe k nearest neighbor classifier produced results for all 4 models in %0.2f minute(s) and %0.2f seconds." % (knn_mins, knn_secs), "\n")

#--------------------------------------------- Naïve Bayes Classification -------------------------------------------#

print("Using Naïve Bayes Classification (Gaussian):\n")
nbc = GaussianNB() # define the naïve bayes classifier using the gaussian algorithm
nbc_start = time.time()

#-------------------- Type-Based Model -------------------#

u2u_p3.classification(nbc, type_model_features, enc_df, fold_count, "Type", targets)

#-------------------- Size-Based Model -------------------#

u2u_p3.classification(nbc, size_model_features, enc_df, fold_count, "Size", targets)

#-------------------- Time-Based Model -------------------#

u2u_p3.classification(nbc, time_model_features, enc_df, fold_count, "Time", targets)

#----------------------- Full Model ----------------------#

u2u_p3.classification(nbc, full_model_features, enc_df, fold_count, "Full", targets)

#---------------------- NBC Run Time ---------------------#

nbc_end = time.time()
nbc_run = (nbc_end - nbc_start)
nbc_mins, nbc_secs = divmod(nbc_run, secs_in_min)
print("\tThe naïve bayes classifier produced results for all 4 models in %0.2f minute(s) and %0.2f seconds." % (nbc_mins, nbc_secs), "\n")

#-------------------------------------------------- END OF SCRIPT ----------------------------------------------------#
