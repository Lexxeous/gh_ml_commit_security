#!/usr/local/bin/python3

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def get_files_grouped_by_hash(validated_df, security_flag):
  flag_df = validated_df[validated_df["SECU_FLAG"] == security_flag]
  files_grouped_by_hash = flag_df.groupby('HASH').agg({'MODIFIED_FILE': list}).reset_index()
  print("First 10", security_flag.lower() ,"modified file name tokens grouped by thier respective commit hash scripts:\n", files_grouped_by_hash.head(10), "\n")
  files_grouped_by_hash.to_csv(security_flag.lower() + r'_hash_files.csv', header=False, index=False)
  print("Shape of", security_flag.lower(), "modified file name tokens grouped by thier commit hash script:", files_grouped_by_hash.shape, "\n")  # INSECURE -> (969, 2); NEUTRAL -> (249984, 2)

  return files_grouped_by_hash


def apply_tfidf(list_of_file_groups):
  tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda doc: doc, preprocessor=lambda doc: doc)
  term_document_matrix = tfidf_vectorizer.fit_transform(list_of_file_groups)
  tfidf_features = np.array(tfidf_vectorizer.get_feature_names())
  print("Length of TF-IDF features:", len(tfidf_features), "\n")  # INSECURE -> 13180; NEUTRAL -> 68495
  print("The resulting TF-IDF term-document matrix:\n", term_document_matrix)
  print("\nShape of the resulting TF-IDF term-document matrix:", term_document_matrix.shape, "\n")  # INSECURE -> (969, 13180); NEUTRAL -> (249984, 68495)

  return tfidf_features, term_document_matrix


def get_top_features(term_document_matrix, tfidf_features):
  coo_matrix = term_document_matrix.tocoo()  # Returns COOrdinate representation of term_document_matrix
  dic = {((row, col), data, tfidf_features[col]) for row, col, data in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)}
  sorted_features = sorted(dic, key=lambda x: x[1], reverse=True)
  top_features = sorted_features[:1000]
  print(top_features)

  return top_features


def write_features_to_file(features, security_flag):
  with open(security_flag.lower() + '_top_features.csv', 'w') as f:
    for row in features:
      f.write(str(row[1]) + ',' + str(row[2]) + '\n')
