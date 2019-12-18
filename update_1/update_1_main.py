#!/usr/local/bin/python3
#coding: utf-8

"""
Title: update_1_main.py
Author(s): Jonathan A. Gibson, Andrew P. Worley, Raunak Shakya
Objectives:
  1. Filter out noisy file names; filenames MUST have extentions and MAY have a path.
  2. Separate dataframes or files into 2 security-related groups: ("INSECURE") and ("NEUTRAL").
  3. Apply text mining (TF-IDF) on the two groups.
  4. Get the text mining matrix, and sort it by TF-IDF scores for both groups.
  5. Take the top 1000 TF-IDF scores for both groups.
  6. Look at the obtained features manually and see what features appear: each member must do
     it individually then discuss agreements and disagreements.
Notes:
  > For the TF-IDF formula, a token ('t') is a unique modified file name and a script ('s') is a unique commit hash.
"""

#--------------------------------------------- Import Necessary Libraries ---------------------------------------------#

import pandas as pd
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utilities")) # add "../utilities" to module import search path
import logger # for custom "Logger" class
import update_1_utils as u1u # for custom "update_2_u1u" utility functions

#-------------------------------------------- Check Command Line Arguments --------------------------------------------#

print("\nProgram Name:", sys.argv[0])
print("Number of command line arguments:", len(sys.argv))
print("The command line argument(s) are:" , str(sys.argv), "\n")

if(len(sys.argv) != 2):
  print("ERROR::160::BAD_ARGUMENTS")
  print("Run program with the following format: \'Python update_1_main.py \"<log_file>\"\' or use \'make run\'.\n")
  sys.exit()

#------------------------------------------------- Load Dataset File -------------------------------------------------#

valid_dataset_names = ["LOCKED_FINAL_CSC4220_5220_DATASET.csv", "dataset.csv", "Dataset.csv", "DATASET.csv", "test.csv", "mini-test.csv"]

for d in valid_dataset_names:
  if(os.path.exists(d)):
    print("Loading dataset file...")
    original_df = pd.read_csv(d)
    print("Done.\n")
    break
  else:
    print("ERROR::002::FILE_NOT_FOUND")
    print("Place project dataset file in root directory with one of the following valid dataset file names:\n", valid_dataset_names, "\n")
    sys.exit()

print("Dataset datatypes:\n", original_df.dtypes, "\n")
print("Dataset original shape:", original_df.shape, "\n") # (7619214, 10)

#------------------------------------------ Filter Out Improper File Names ------------------------------------------#

print("Filtering out improper modified file names...\n")
validated_file_pattern = r'^\/?[\w_\-\/\"\s]+\.\w\w?\w?\w?$'
validated_df = original_df[original_df["MODIFIED_FILE"].str.contains(validated_file_pattern, case=False, na=False, regex=True)]
print("Dataset filtered shape:", validated_df.shape, "\n")  # (1557292, 10)
validated_file_names = validated_df['MODIFIED_FILE']
validated_file_names.to_csv(r'validated_file_names.csv', header=False, index=False)

#---------------------------------------------- Apply TF-IDF Text Mining --------------------------------------------#

for security_flag in ['INSECURE', 'NEUTRAL']:
  print('\n***** START OF ANALYSIS FOR', security_flag, 'SECURITY FLAG *****\n')
  print("Grouping", security_flag.lower(), "modified file name tokens by thier respective commit hash scripts...\n")
  files_grouped_by_hash_df = u1u.get_files_grouped_by_hash(validated_df, security_flag)
  list_of_file_groups = files_grouped_by_hash_df['MODIFIED_FILE'].tolist()  # [['a.c'], ['b.py', 'c.cpp'], ....]
  print("Applying TF-IDF text mining algorithm...\n")
  tfidf_features, term_document_matrix = u1u.apply_tfidf(list_of_file_groups)
  print("Sorting", security_flag.lower(), "token scores in decending order...\n")
  top_features = u1u.get_top_features(term_document_matrix, tfidf_features)
  print("\nWriting the top 1000", security_flag.lower(), "modified file token scores to file...")
  u1u.write_features_to_file(top_features, security_flag)
  print('\n***** END OF ANALYSIS FOR', security_flag, 'SECURITY FLAG *****\n')

#-------------------------------------------------- END OF SCRIPT ---------------------------------------------------#
print("#-------------------------------------------------- END OF SCRIPT ---------------------------------------------------#")
