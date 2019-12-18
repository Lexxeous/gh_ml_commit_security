#!/usr/local/bin/python3
# coding: utf-8

'''
Filename: update_3_main_p3.py
Authors: Jonathan A. Gibson, Andrew P. Worley, & Raunak S. Hakya
Objectives:
	Phase 0:
		Item 0:
			Identify and partition dataset groups: "ComputationalChemistry", "Astronomy", "ComputationalBiology", and "Others".
			"OTHERS" will include "ComputationalScience" & "ComputationalIntelligence".
		Item 1:
			For each dataset (partitioned dataframe), use Mann Whitney U Tests and Cliff's Effect Size to find differences between "INSECURE" & "NEUTRAL" "SECU_FLAG" labels.
	Phase 1:
		Item 2:
			Apply full model: (ADD_LOC, DEL_LOC, TOT_LOC,DEV_EXP, DEV_RECENT, PRIOR_AGE, CHANGE_FILE_CNT).
			Build model for each type of dataset, then use one dataset to train and test against others.
			Represent the following 12 (Train, Test) pairs for each of the 5 classification models (60 result sets).
			[
				[(Chemistry, Astronomy), (Chemistry ,Biology), (Chemistry, Others)],
				[(Astronomy, Biology), (Astronomy, Chemistry), (Astronomy, Others)],
				[(Biology, Astronomy), (Biology, Chemistry), (Biology, Others)],
				[(Others, Biology), (Others, Astronomy), (Others, Chemistry)]
			]
		Item 3:
			Apply deep neural network to compare against classification algorithms, for each pair.
	Phase 2:
		Item 4:
			Apply SMOTE on training datasets for each pair, and then build models with all learners.
		Item 5:
			Report confusion matrix, precision, recall, and F-measure for Item 2, Item 3, and Item 4.
			Report on Items 3 & 4 should be a "before and after" result based on the lack and use of SMOTE sampling respectively.
	Phase 3:
		Item 6:
			Run security static analysis tool on code to find security weaknesses.
			Use "Bandit" library to report severity.

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

	SMOTE Reference(s):
		https://imbalanced-learn.readthedocs.io/en/stable/api.html
		https://www.kaggle.com/qianchao/smote-with-imbalance-data
			ENN (Edited Nearest Neighbors) Cleaning vs. Tomek Links Cleaning
				Cleaning techniques remove the overlapping that is introduced from sampling methods.
				Majority class examples might be invading the minority class space and vice-versa.
				ENN tends to remove more examples than the Tomek links does, so it is expected that it will provide a more in depth data cleaning.
				Any example that is misclassified by ENN's three nearest neighbors is removed from the training set.
				The following paper proposed SMOTE+ENN & SMOTE+Tomek and showed that both worked very well, especially SMOTE+ENN.
					http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.7757&rep=rep1&type=pdf

	Deep Learning (Keras) Reference(s):
		https://keras.io
		https://machinelearningmastery.com/tutorial-first-neural-network-python-keras
		https://datascience.stackexchange.com/questions/51100/keras-how-to-connect-a-cnn-model-with-a-decision-tree
		https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

	Bandit Reference(s):
		https://pypi.org/project/bandit/
'''

#--------------------------------------------- Import Necessary Libraries ---------------------------------------------#

from sklearn.tree import DecisionTreeClassifier # for decision tree classification
from sklearn.ensemble import RandomForestClassifier # for random forest classification
from sklearn.neural_network import MLPClassifier # artificial neural network classification
from sklearn.neighbors import KNeighborsClassifier # for k nearest neighbor classification
from sklearn.naive_bayes import GaussianNB # for naïve bayes classification
import sklearn.metrics as metric # for printing a classification report and/or confusion matrix
from keras.models import Sequential # for building Keras DNN model
from keras.layers import Dense # for building Keras DNN model
import pandas as pd # for Dataframe support
import numpy as np # for mathmatical support
import os.path # for testing file existance
import sys # for managing command line arguments

#------------------------------------------- Import Custom Utility Modules -------------------------------------------#

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utilities")) # add "../utilities" to module import search path
import logger # for custom "Logger" class
import update_3_utils_p3 as u3u_p3 # for custom "update_3_utils" utility functions

#-------------------------------------------- Check Command Line Arguments --------------------------------------------#

print("\nProgram Name: ", sys.argv[0])
print("Number of command line arguments:", len(sys.argv))
print("The command line argument(s) are:" , str(sys.argv), "\n")

if(len(sys.argv) != 2):
	print("ERROR::160::BAD_ARGUMENTS")
	print("Run program with the following format: \'python update_3_main_p#.py \"<log_file>\"\'; alternatively use \'make run2\' or \'make run3\' for Python2 and Python3 appropriate scripts respectively.\n")
	sys.exit()

#------------------------------------------------- Load Dataset File -------------------------------------------------#

valid_dataset_names = ["update_3_dataset.csv", "dataset.csv", "Dataset.csv", "DATASET.csv", "test.csv"]

for d in valid_dataset_names:
	if(os.path.exists(d)):
		print("Loading dataset file...")
		df = pd.read_csv(d)
		print("Done.\n")
		break
	else:
		print("ERROR::002::FILE_NOT_FOUND")
		print("Place project dataset file in root directory with one of the following valid dataset file names:\n", valid_dataset_names, "\n")
		sys.exit()

#------------------------------------------------------- Setup -------------------------------------------------------#

skip = True # option to skip larger datasets for faster testing/debgging

dnn_epochs = 50
dnn_batch_sz = 10
fit_verbose = 2 # 0 for silent, 1 for full, 2 for minimal
pred_verbose = 0 # 0 for silent, 1 for full

target_df, targets = u3u_p3.encode_target(df, "SECU_FLAG") # encoding dataframe with "SECU_FLAG" as classification target
targets_dict = u3u_p3.col_count_perc(target_df, "SECU_FLAG", len(df.index), 2) # count unique targets and calculate percentiles

full_model_features = list(target_df.columns[3:10]) # ADD_LOC, DEL_LOC, TOT_LOC, DEV_EXP, DEV_RECENT, PRIOR_AGE, CHANGE_FILE_CNT
single_type_groups = ["ComputationalChemistry", "Astronomy", "ComputationalBiology"] # array for only single repository type dataset groups
all_groups = ["ComputationalChemistry", "Astronomy", "ComputationalBiology", "Others"] # array for all 4 dataset groups
model_classifiers = ["DTC", "RFC", "KNN", "ANN", "NBC"] # array for the 5 prior learners (classifier models)

N = len(all_groups) # N = 4
M = len(model_classifiers) # M = 5

classifiers = [DecisionTreeClassifier(random_state=1),
							 RandomForestClassifier(n_estimators=50, random_state=2),
							 KNeighborsClassifier(n_neighbors=3),
							 MLPClassifier(random_state=3),
							 GaussianNB()]

org_model_set_matrix = [ [ 0 for i in range(M) ] for j in range(N) ] # original matrix of size 4x5 (NxM) for [dataset_grouping][classifier_model]
res_model_set_matrix = [ [ 0 for i in range(M) ] for j in range(N) ] # resampled matrix of size 4x5 (NxM) for [dataset_grouping][classifier_model]

# Contents of <XYZ_model_set_matrix>:
# |---------0---------|---------1---------|---------2---------|---------3---------|
# |--------CHM--------|--------AST--------|--------BIO--------|--------OTH--------|
# |DTC|RFC|KNN|ANN|NBC|DTC|RFC|KNN|ANN|NBC|DTC|RFC|KNN|ANN|NBC|DTC|RFC|KNN|ANN|NBC|
# |-0-|-1-|-2-|-3-|-4-|-0-|-1-|-2-|-3-|-4-|-0-|-1-|-2-|-3-|-4-|-0-|-1-|-2-|-3-|-4-|

# Define the Keras model layers
model = Sequential()
model.add(Dense(12, input_dim=len(full_model_features), activation="relu")) # dense input layer, 12 neurons and 7 features
model.add(Dense(len(full_model_features), activation="relu")) # dense input layer, 7 neurons
model.add(Dense(1, activation="sigmoid")) # prediction output layer

# Compile the Keras model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

print("\nEncoded Targets for Entire Dataset:")
for i in range(len(targets)): # print "NEUTRAL" and "INSECURE" statistics for the entire dataset
	print("\t", targets[i], "–> Encoded as:", str(i)+',',\
													"Count:", int(targets_dict[i][0]), "/ 300056,",\
													"Percentage:", str(targets_dict[i][1]*100)+"%")

#--------------------------------------- Spilt Dataset into Partitioned Groups ---------------------------------------#

datasets = u3u_p3.partition_df_repo_type_groups(target_df, single_type_groups) # [chemistry_df, astronomy_df, biology_df, others_df]

# Loop through all 4 of the partitioned dataset groups
for i in range(N): # N = 4
	dataset_dict = u3u_p3.col_count_perc(datasets[i], "SECU_FLAG", len(datasets[i].index), 2) # calculate percentitles for partitioned dataframes
	print("\nEncoded Targets for Original " + all_groups[i] + " Dataset:")
	for j in range(len(targets)): # print "NEUTRAL" and "INSECURE" statistics for each partitioned dataset group
		print("\t", targets[j], "–> Encoded as:", str(j)+',',\
														"Count:", int(dataset_dict[j][0]), "/", str(len(datasets[i].index))+',',\
														"Percentage:", str(dataset_dict[j][1]*100)+"%")

#--------------------------------- Apply Mann Whitney U Tests and Cliff's Effect Size --------------------------------#

# Loop through all 4 of the partitioned dataset groups
for g in range(len(datasets)):
	
	u3u_p3.print_parent_divider("effect", g, all_groups) # formatting printed output

	if(skip):
		if(g in [1,2,3]): continue

	# Loop through all 6 of the features in the full model
	for f in range(len(full_model_features)):

		# Create 2 generic lists to compare feature values based on "NEUTRAL" and "INSECURE" divisions
		group_feat_n_list = []
		group_feat_i_list = []

		for idx, row in datasets[g].iterrows(): # loop through all the rows in dataset group 'g'

			if(row["SECU_FLAG"] == 0):
				group_feat_n_list.append(row[full_model_features[f]])
			else:
				group_feat_i_list.append(row[full_model_features[f]])

		print("\nComparing \"NEUTRAL\" and \"INSECURE\" lists, using the " + all_groups[g] + " dataset, for the \"" + full_model_features[f] + "\" feature:")
		u3u_p3.compare_lists(group_feat_n_list, group_feat_i_list, u3u_p3.CD_LRG_DIFF)

#---------------------------------- Train/Build Full Model for Each Original Dataset ---------------------------------#

u3u_p3.print_parent_divider("fit") # formatting printed output

# Loop through dataset groupings
for i in range(N): # N = 4
	model_list = [None]*5

	subset_data_X = datasets[i][full_model_features]
	class_labels_Y = datasets[i]["SECU_FLAG"]
	print("\n")

	# Loop through classification models
	for j in range(M): # M = 5
	
		model_list[j] = classifiers[j]
		model_list[j].fit(subset_data_X, class_labels_Y)
		org_model_set_matrix[i][j] = model_list[j]
		print("Finished building \"" + model_classifiers[j] + "\" classifier model for original \"" + all_groups[i] + "\" dataset.")

#---------------------- Test Each Original Dataset Against Others (12 Pairs Per Classifier Model) --------------------#

# Loop through dataset groupings (hosts)
for i in range(N): # N = 4

	u3u_p3.print_parent_divider("prior", i, all_groups) # formatting printed output

	# Loop through dataset groupings again (guests)
	for j in range(N): # N = 4
		if(i == j): continue # dataset group shouldn't compare against itself
    
		test_set = datasets[j][full_model_features] # get the dataset for the current guest
		actual = datasets[j]["SECU_FLAG"] # get the ground truth for the current guest

		# Loop through prior classifier models
		for k in range(M): # M = 5
			u3u_p3.print_child_divider(model_classifiers[k]) # formatting printed output
			prediction = org_model_set_matrix[i][k].predict(test_set)
		  
		  # Skipping datasets for testing purposes
			if(skip):
				if(k in [1,2,3]): continue

			host, guest = u3u_p3.set_host_and_guest(i, j, single_type_groups) # set proper host and guest

			#------------------------------------ Results with Prior Learners (BEFORE) -------------------------------------#

			print("Results for \"" + model_classifiers[k] + "\" trained on original \"" + host + "\" dataset tested with original \"" + guest + "\" dataset:")
			print(metric.classification_report(actual, prediction, target_names=targets))
			print("Confusion Matrix:")
			print(metric.confusion_matrix(actual, prediction))

#-------------------------------------- Results with DNN for Original Datasets ---------------------------------------#

print("\nUsing", dnn_epochs, "epoch(s) and a batch size of", str(dnn_batch_sz) + "...")

# Loop through dataset groupings (hosts)
for i in range(N): # N = 4
	
	u3u_p3.print_parent_divider("dnn", i, all_groups) # formatting printed output

	# Skipping datasets for testing purposes
	if(skip):
		if(i != 3): continue

	# Fit the Keras DNN model once for each of the 4 datasets
	model.fit(datasets[i][full_model_features], datasets[i]["SECU_FLAG"], epochs=dnn_epochs, batch_size=dnn_batch_sz, verbose=fit_verbose)

	# Loop through dataset groupings again (guests)
	for j in range(N): # N = 4
		if(i == j): continue # dataset group shouldn't compare against itself
	
		host, guest = u3u_p3.set_host_and_guest(i, j, single_type_groups)

		# Test the fit Keras DNN model with other the other datasets
		pred_mat = model.predict(datasets[j][full_model_features], verbose=pred_verbose)
		ground_truth = datasets[j]["SECU_FLAG"]
		acc, prec, rec, fscore = u3u_p3.get_dnn_pred_metrics(ground_truth, pred_mat)
		true_arr, pred_arr = u3u_p3.process_dfcol_and_mat(ground_truth, pred_mat)

		print("\nResults for Deep Neural Network trained on original \"" + host + "\" dataset tested with original \"" + guest + "\" dataset:")
		print("\tAccuracy: " + str(round(acc*100, 2)) + "%, Precision: " + str(prec) + ", Recall: " + str(rec) + ", F-Score: " + str(fscore))
		print("\tConfusion Matrix:")
		print(metric.confusion_matrix(true_arr, pred_arr))

#----------------------------------------------- Resampling with SMOTE -----------------------------------------------#

resampled_datasets = [None]*len(datasets)
for d in range(len(datasets)):
	resampled_datasets[d] = u3u_p3.resample_dataset(datasets[d], full_model_features, all_groups[d])

# Loop through all 4 of the resampled, partitioned dataset groups
for i in range(N): # N = 4
	resampled_dataset_dict = u3u_p3.col_count_perc(resampled_datasets[i], "SECU_FLAG", len(resampled_datasets[i].index), 2) # calculate percentitles for resampled dataframes
	print("\nEncoded Targets for Resampled " + all_groups[i] + " Dataset:")
	for j in range(len(targets)): # print "NEUTRAL" and "INSECURE" statistics for each partitioned dataset group
		print("\t", targets[j], "–> Encoded as:", str(j)+',',\
														"Count:", int(resampled_dataset_dict[j][0]), "/", str(len(resampled_datasets[i].index))+',',\
														"Percentage:", str(resampled_dataset_dict[j][1]*100)+"%")

#--------------------------------- Train/Build Full Model for Each Resampled Dataset  --------------------------------#

u3u_p3.print_parent_divider("fit") # formatting printed output

# Loop through dataset groupings
for i in range(N): # N = 4
	model_list = [None]*5

	subset_data_X = resampled_datasets[i][full_model_features]
	class_labels_Y = resampled_datasets[i]["SECU_FLAG"]
	print("\n")

	# Loop through classification models
	for j in range(M): # M = 5
	
		model_list[j] = classifiers[j]
		model_list[j].fit(subset_data_X, class_labels_Y)
		res_model_set_matrix[i][j] = model_list[j]
		print("Finished building \"" + model_classifiers[j] + "\" classifier model for resampled \"" + all_groups[i] + "\" dataset.")

#--------------------- Test Each Resampled Dataset Against Others (12 Pairs Per Classifier Model) --------------------#

# Loop through dataset groupings (hosts)
for i in range(N): # N = 4

	u3u_p3.print_parent_divider("prior", i, all_groups) # formatting printed output

	# Loop through dataset groupings again (guests)
	for j in range(N): # N = 4
		if(i == j): continue # dataset group shouldn't compare against itself
    
		test_set = resampled_datasets[j][full_model_features] # get the dataset for the current guest
		actual = resampled_datasets[j]["SECU_FLAG"] # get the ground truth for the current guest

		# Loop through prior classifier models
		for k in range(M): # M = 5
			u3u_p3.print_child_divider(model_classifiers[k]) # formatting printed output
			prediction = res_model_set_matrix[i][k].predict(test_set)
		  
		  # Skipping datasets for testing purposes
			if(skip):
				if(k in [1,2,3]): continue

			host, guest = u3u_p3.set_host_and_guest(i, j, single_type_groups) # set proper host and guest

			#------------------------------------- Results with Prior Learners (AFTER) -------------------------------------#

			print("Results for \"" + model_classifiers[k] + "\" trained on resampled \"" + host + "\" dataset, tested with resampled \"" + guest + "\" dataset:")
			print(metric.classification_report(actual, prediction, target_names=targets))
			print("Confusion Matrix:")
			print(metric.confusion_matrix(actual, prediction))

#-------------------------------------- Results with DNN for Resampled Datasets --------------------------------------#

print("\nUsing", dnn_epochs, "epoch(s) and a batch size of", str(dnn_batch_sz) + "...")

# Loop through dataset groupings (hosts)
for i in range(N): # N = 4
	
	u3u_p3.print_parent_divider("dnn", i, all_groups) # formatting printed output

	# Skipping datasets for testing purposes
	if(skip):
		if(i != 3): continue

	# Fit the Keras DNN model once for each of the 4 datasets
	model.fit(resampled_datasets[i][full_model_features], resampled_datasets[i]["SECU_FLAG"], epochs=dnn_epochs, batch_size=dnn_batch_sz, verbose=fit_verbose)

	# Loop through dataset groupings again (guests)
	for j in range(N): # N = 4
		if(i == j): continue # dataset group shouldn't compare against itself
	
		host, guest = u3u_p3.set_host_and_guest(i, j, single_type_groups)

		# Test the fit Keras DNN model with other the other datasets
		pred_mat = model.predict(resampled_datasets[j][full_model_features], verbose=pred_verbose)
		ground_truth = resampled_datasets[j]["SECU_FLAG"]
		acc, prec, rec, fscore = u3u_p3.get_dnn_pred_metrics(ground_truth, pred_mat)
		true_arr, pred_arr = u3u_p3.process_dfcol_and_mat(ground_truth, pred_mat)

		print("\nResults for Deep Neural Network trained on resampled \"" + host + "\" dataset tested with resampled \"" + guest + "\" dataset:")
		print("\tAccuracy: " + str(round(acc*100, 2)) + "%, Precision: " + str(prec) + ", Recall: " + str(rec) + ", F-Score: " + str(fscore))
		print("\tConfusion Matrix:")
		print(metric.confusion_matrix(true_arr, pred_arr))

#-------------------------------------------------- END OF SCRIPT ----------------------------------------------------#
