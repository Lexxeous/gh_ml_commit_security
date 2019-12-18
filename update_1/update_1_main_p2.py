# coding: utf-8

"""
Title: update_1_main.py
Author(s): Jonathan A. Gibson, Andrew P. Worley, Raunak Shakya
Objectives:
  1. Filter out noisy file names; filenames MUST have extentions and MAY have a path.
  2. Separate dataframes or files into 2 security-related groups: ("INSECURE") and ("NEUTRAL").
  3. Apply text mining (TF-IDF) on the two groups.
  4. Get the text mining matrix, and sort it by TF-IDF scores for both groups.
  5. Take the top N TF-IDF scores for both groups.
  6. Look at the obtained features manually and see what features appear.
Notes:
  > For the TF-IDF formula, a token ('t') is a unique modified file name and a script ('s') is a unique commit hash.
"""

#--------------------------------------------- Import Necessary Libraries --------------------------------------------#

import pandas as pd # for dataframe support
import os.path
import sys

#------------------------------------------- Import Custom Utility Modules -------------------------------------------#

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utilities")) # add "../utilities" to module import search path
import logger # for custom "Logger" class
import update_1_utils_p2 as u1u_p2 # for custom "update_1_utils_p3" utility functions

#-------------------------------------------- Check Command Line Arguments -------------------------------------------#

print "\nProgram Name:", sys.argv[0]
print "Number of command line arguments:", len(sys.argv)
print "The command line argument(s) are:", str(sys.argv)

if(len(sys.argv) != 2):
  print "ERROR::160::BAD_ARGUMENTS"
  print "Run program with the following format: \'python update_1_main.py \"<log_file>\"\'; alternatively use \'make run2\' or \'make run3\' for Python2 and Python3 appropriate scripts respectively.\n"
  sys.exit()

#------------------------------------------------- Load Dataset File -------------------------------------------------#

valid_dataset_names = ["update_1_dataset.csv", "dataset.csv", "Dataset.csv", "DATASET.csv"]

for d in valid_dataset_names:
  if(os.path.exists(d)):
    print "\nLoading dataset file..."
    original_df = pd.read_csv(d)
    print "Done."
    break
  else:
    print "ERROR::002::FILE_NOT_FOUND"
    print "Place project dataset file in root directory with one of the following valid dataset file names:\n", valid_dataset_names, "\n"
    sys.exit()

print "\nDataset datatypes:"
print original_df.dtypes

print "\nDataset original shape:", original_df.shape # (7619214, 10)

#------------------------------------------ Filter Out Improper File Names -------------------------------------------#

print "\nFiltering out improper modified file names..."
validated_file_pattern = r"^\/?[\w_\-\/\"\s]+\.\w\w?\w?\w?$"
validated_df = original_df[original_df["MODIFIED_FILE"].str.contains(validated_file_pattern, case=False, na=False, regex=True)]

print "\nDataset filtered shape:", validated_df.shape # (1557292, 10)
validated_file_names = validated_df['MODIFIED_FILE']
validated_file_names.to_csv(r"validated_file_names.csv", header=False, index=False)

#---------------------------------------------- Apply TF-IDF Text Mining ---------------------------------------------#

for security_flag in ["INSECURE", "NEUTRAL"]:
  print "\n************************** START OF ANALYSIS FOR", security_flag, "SECURITY FLAG **************************"

  print "\nGrouping", security_flag, "modified file name tokens by thier respective commit hash scripts..."
  files_grouped_by_hash_df = u1u_p2.get_files_grouped_by_hash(validated_df, security_flag)
  list_of_file_groups = files_grouped_by_hash_df['MODIFIED_FILE'].tolist()  # [['a.c'], ['b.py', 'c.cpp'], ....]

  print "\nApplying TF-IDF text mining algorithm..."
  tfidf_features, term_document_matrix = u1u_p2.apply_tfidf(list_of_file_groups)

  print "\nSorting", security_flag, "token scores in decending order..."
  top_features = u1u_p2.get_top_features(term_document_matrix, tfidf_features)

  print "\nWriting the top 1000", security_flag, "modified file token scores to file..."
  u1u_p2.write_features_to_file(top_features, security_flag)

  print "\n************************** END OF ANALYSIS FOR", security_flag, "SECURITY FLAG **************************"

#-------------------------------------------------- END OF SCRIPT ----------------------------------------------------#
