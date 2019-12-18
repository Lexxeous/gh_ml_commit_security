#!/usr/local/bin/python3
# coding: utf-8

#--------------------------------------------- Import Necessary Libraries --------------------------------------------#

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#-------------------------------------------------- Global Variables -------------------------------------------------#

N = 1000

#---------------------------------------------- Utilities Implementation ---------------------------------------------#

''' Takes a filtered/validated dataframe and a single security flag as input then returns unique hashes and a list of associated filenames for each unique hash. '''
def get_files_grouped_by_hash(validated_df, security_flag):
  flag_df = validated_df[validated_df["SECU_FLAG"] == security_flag]
  files_grouped_by_hash = flag_df.groupby("HASH").agg({"MODIFIED_FILE": list}).reset_index()
  print("\nFirst 10", security_flag, "modified file name tokens grouped by thier respective commit hash scripts:\n", files_grouped_by_hash.head(10))
  files_grouped_by_hash.to_csv(security_flag.lower() + r"_hash_files.csv", header=False, index=False)
  print("\nShape of", security_flag, "modified file name tokens grouped by thier commit hash script:", files_grouped_by_hash.shape) # INSECURE -> (969, 2); NEUTRAL -> (249984, 2)

  return files_grouped_by_hash


''' Takes a list of lists where each element is a set of modified files for each unique hash then returns the TF-IDF feature tuples and a term-document matrix. '''
def apply_tfidf(list_of_file_groups):
  tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda doc: doc, preprocessor=lambda doc: doc)
  term_document_matrix = tfidf_vectorizer.fit_transform(list_of_file_groups)
  tfidf_features = np.array(tfidf_vectorizer.get_feature_names())
  print("\nLength of TF-IDF features:", len(tfidf_features)) # INSECURE -> 13180; NEUTRAL -> 68495
  print("\nThe resulting TF-IDF term-document matrix:")
  print(term_document_matrix)
  print("\nShape of the resulting TF-IDF term-document matrix:", term_document_matrix.shape) # INSECURE -> (969, 13180); NEUTRAL -> (249984, 68495)

  return tfidf_features, term_document_matrix


''' Takes a term-document matrix and TF-IDF features, sorts the features in decending order, and returns the top N features. '''
def get_top_features(term_document_matrix, tfidf_features):
  coo_matrix = term_document_matrix.tocoo() # returns COOrdinate representation of term_document_matrix
  dic = {((row, col), data, tfidf_features[col]) for row, col, data in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)}
  sorted_features = sorted(dic, key=lambda x: x[1], reverse=True)
  top_features = sorted_features[:N]
  print(top_features)

  return top_features


''' Takes a list of features and a security flag and writes the scores/features to a file called "<security_flag>_top_features.csv". '''
def write_features_to_file(features, security_flag):
  with open(security_flag.lower() + '_top_features.csv', 'w') as f:
    for row in features:
      f.write(str(row[1]) + ',' + str(row[2]) + '\n')

#---------------------------------------------------------------------------------------------------------------------#
