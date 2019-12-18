# coding: utf-8

#--------------------------------------------- Import Necessary Libraries --------------------------------------------#

from __future__ import division
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from imblearn.combine import SMOTEENN # latest version for Python2 is imbalanced-learn==0.4.2
from keras import backend as K
import statistics # for calculating mean, median, and standard deviation
from scipy import stats
from tqdm import tqdm # for progress bar usage
import pandas as pd
import numpy as np
import warnings # for managing warnings
import sys
import math

#-------------------------------------------------- Global Variables -------------------------------------------------#

CD_SML_DIFF = 0.147
CD_MED_DIFF = 0.330
CD_LRG_DIFF = 0.474

NEUTRAL = 0
INSECURE = 1

div_width = 150

#------------------------------------------------------- Setup -------------------------------------------------------#

warnings.filterwarnings("ignore") # ignore warnings ; for 0 valued F1 scores

#---------------------------------------------- Utilities Implementation ---------------------------------------------#

''' Add column to dataframe with encoded integers for the classification column target. '''
def encode_target(df, target_column):
  df_mod = df.copy() # modified dataframe
  targets = df_mod[target_column].unique() # NEUTRAL & INSECURE
  map_to_int = {name: n for n, name in enumerate(targets)} # [NEUTRAL, INSECURE] => [0, 1]
  df_mod[target_column] = df_mod[target_column].replace(map_to_int)
  print "Dataset after encoding the target column:", "\n", df_mod.head(), "\n"
  return (df_mod, targets)


''' Count unique column values and calculate individual percentiles. '''
def col_count_perc(df, col, num_rows, precision):
  unique_names = df[col].unique()
  dictionary = {} # initialize empty dictionary
  cnt = 0
  perc = 1

  for name in unique_names: # create dictionary size of len(<unique_names>)
    dictionary[name] = [0.0, 0.0] # [cnt, perc]

  for row in df[col]:
    for name in unique_names:
      if(row == name): dictionary[name][cnt] += 1 # get count for all names in <dictionary>

  for name in unique_names:
    dictionary[name][perc] = round(dictionary[name][cnt] / num_rows, precision) # calculate percentile for each key-value pair

  return dictionary


''' Replace a dataframe column that has string values with an integer encoded column. '''
def encode_string_column(df, string_column):
  df_mod = df.copy() # modified dataframe
  targets = df_mod[string_column].unique() # "ComputationalChemistry, Astronomy, ..."
  map_to_int = {name: n for n, name in enumerate(targets)} # [NEUTRAL, INSECURE] => [0, 1]
  df_mod[string_column] = df_mod[string_column].replace(map_to_int)
  print "Dataset after encoding the string column:", "\n", df_mod.head(), "\n"
  return (df_mod, targets)


''' Count and print the quantity of True Negatives, False Negatives, False Positives, and True Positives '''
def count_true_false_neg_pos(true, pred):
  tn = fn = fp = tp = 0
  for i in range(0, len(pred)):
    if (true[i] == 0 and pred[i] == 0): tn += 1
    elif (true[i] == 0 and pred[i] == 1): fn += 1
    elif (true[i] == 1 and pred[i] == 0): fp += 1
    elif (true[i] == 1 and pred[i] == 1): tp += 1
    else:
      sys.exit("ERROR::004::COUNTING_ERROR")

  # print "\tTrue Negatives: " + str(tn)
  # print "\tFalse Negatives: " + str(fn)
  # print "\tFalse Positives: " + str(fp)
  # print "\tTrue Positives: " + str(tp) + "\n"
  return tn, fn, fp, tp


''' Process the prediction output from the Keras model DNN with ground truth DataFrame_Column::<actual> and prediction Matrix::<pred_mat>. '''
def get_dnn_pred_metrics(actual_dfcol, pred_mat):
  true_arr, pred_arr = process_dfcol_and_mat(actual_dfcol, pred_mat)
  tn, fn, fp, tp = count_true_false_neg_pos(true_arr, pred_arr)
  acc, prec, rec, fscore = calc_dnn_pred_metrics(tn, fn, fp, tp)
  return acc, prec, rec, fscore


''' Process the ground truth dataframe column and the DNN prediction matrix and return them both as lists. '''
def process_dfcol_and_mat(actual_dfcol, pred_mat):
  pred_arr = []
  true_arr = []

  # Build list of ground truth values
  for a in actual_dfcol:
    true_arr.append(a)

  # Build list of prediction values
  for p in pred_mat:
    if(p[0] < 0.5):
      pred_arr.append(NEUTRAL)
    else:
      pred_arr.append(INSECURE)

  return true_arr, pred_arr


''' Use equations to calculate accuracy, precision, recall, and f-score ; also account for special cases. '''
def calc_dnn_pred_metrics(tn, fn, fp, tp):
  special = False

  # Account for special cases
  if(tp == 0 and fp == 0 and fn == 0):
    prec = rec = fscore = 1
    special = True
  elif(tp == 0 and (fp > 0 or fn > 0)):
    prec = rec = fscore = 0
    special = True

  # If special cases dont apply, use default equations
  if(not special):
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fscore = (2*prec*rec)/(prec+rec)

  acc = (tp+tn)/(tn+fn+fp+tp)
  return acc, round(prec, 2), round(rec, 2), round(fscore, 2)


''' Partition original dataframe into 4 grouped dataframes based on "REPO_TYPE". Multi-type group (<stg>) dataframe returned at the end of <types_df>. '''
def partition_df_repo_type_groups(df, stg):
  type_dfs = [None]*(len(stg)+1) # create empty dataframe list
  remainder = df # others dataframe to filter out unwanted repo types

  for i in range(0, len(stg)):
    type_dfs[i] = df[df.REPO_TYPE == stg[i]]
    remainder = remainder[remainder.REPO_TYPE != stg[i]] # filtering out unwanted repo types
    if(i == len(stg)-1): # on last iteration
      type_dfs[i+1] = remainder # add the filtered (remainder) dataframe to the list of dataframes, partitioned by "REPO_TYPE" groups
  return type_dfs


''' Resamples with SMOTEENN and reduces <df> based on the given feature colums from <feature_list> ; re-initializes "SECU_FLAG" column and <repo_type> column. '''
def resample_dataset(df, feature_list, repo_type):
  num_rows = len(df.index) # number of rows in <df>
  num_features = len(feature_list) # number of feature columns to resample
  cur_row = [] # list to hold the current row of <df>
  feat_val_mat = [] # the matrix (list of lists) to hold all feature values
  counter = 0 # counter for progress

  print "\nResampling data for the " + repo_type + " dataset..."
  for idx, row in tqdm(df.iterrows(), desc="\tProgress"): # loop <num_rows> times
    counter += 1
    # print_progress(counter, num_rows)
    for j in range(num_features): # loop <num_features> times
      cur_row.append(row[feature_list[j]]) # form list of current row values
    feat_val_mat.append(cur_row) # append <cur_row> to <feat_val_mat>
    cur_row = []

  smote_obj = SMOTEENN(sampling_strategy="all", random_state=99) # <smote_obj> should over/under-sample both the "NEUTRAL" and "INSECURE" classes
  resampled_data, resampled_targets = smote_obj.fit_resample(feat_val_mat, list(df["SECU_FLAG"]))

  resampled_df = pd.DataFrame(resampled_data, columns=feature_list) # recreate the reduced dataframe
  resampled_df["SECU_FLAG"] = resampled_targets # re-initialize the "SECU_FLAG" column
  resampled_df["REPO_TYPE"] = [repo_type]*len(resampled_df.index) # re-initialize the "REPO_TYPE" column
  return resampled_df


''' Print a parent formatted divider with type String::<alg>, index Integer::<idx>, dataset groups List::<all_groups>, and width <width> '''
def print_parent_divider(alg, idx=-1, all_groups=[], width=div_width):
  if(alg.upper() == "DNN"):
    s = "Training Deep Neural Network with " + all_groups[idx] + " Dataset for Transfer Learning"
  elif(alg.upper() == "PRIOR"):
    s = "Testing Prior Classifiers against " + all_groups[idx] + " Dataset for Transfer Learning"
  elif(alg.upper() == "EFFECT"):
    s = "Comparing " + all_groups[idx] + " Features"
  elif(alg.upper() == "FIT"):
    s = "Fitting Prior Learner Classification Models for Partitioned Datasets"
  else:
    s1 = ""

  right = int(math.ceil((width - len(s))/2))
  left = int(math.floor((width - len(s))/2))
  print "\n", '='*left, s, '='*right


''' Print a child formatted divider with string <s> and a customizable divider width <width> '''
def print_child_divider(s, width=div_width):
  right = int(math.ceil((width - len(s))/2))
  left = int(math.floor((width - len(s))/2))
  print "\n", '*'*left, s, '*'*right


''' Set the host and guest variables for transfer learning pairs with parent index <idx_i>, child index <idx_j>, and single type groups List::<stg> '''
def set_host_and_guest(idx_i, idx_j, stg):
  # Set proper host dataset training group
  if(idx_i < len(stg)): host = stg[idx_i]
  else: host = "Others"

  # Set proper guest dataset testing group
  if(idx_j < len(stg)): guest = stg[idx_j]
  else: guest = "Others"

  return host, guest


''' Return a boolean based on rounded Cliff's Delta (CD) function value and the <diff> parameter, and the CD value. '''
def cliffs_delta(lst1, lst2, diff): # differences are [small, medium, large][0]
  m, n = len(lst1), len(lst2)
  lst2 = sorted(lst2)
  j = more = less = 0
  for repeats, x in runs(sorted(lst1)):
    while j <= (n - 1) and lst2[j] <  x:
      j += 1
    more += j*repeats
    while j <= (n - 1) and lst2[j] == x:
      j += 1
    less += (n - j)*repeats
  d = (more - less) / (m*n)
  return abs(d) > diff, round(abs(d), 3) # returns true if there are more than <diff> differences and the calacluated cliffs delta value


''' Iterator, chunks repeated values. '''
def runs(lst):
  for j, two in enumerate(lst):
    if(j == 0):
      one, i = two, 0
    if(one != two):
      yield j - i, one
      i = j
    one = two
  yield j - i + 1, two


''' Demo function for Cliff's Delta. '''
def cliffs_delta_demo():
  lst1=[1, 2, 3, 4, 5, 6, 7]
  for r in [1.01, 1.1, 1.21, 1.5, 2]:
    lst2=map(lambda x: x*r,lst1)
    print r, cliffs_delta(lst1,lst2) # should return False


''' Takes defective and non-defective values for feature and prints the median, mean, and count with the feature, p-value, and Cliff's delta '''
def compare_lists(i_lst, n_lst, diff):
  cd = None
  print "\tNeutral values (MEDIAN): {}, (MEAN): {}, (COUNT): {}".format(np.median(list(n_lst)), np.mean(list(n_lst)), len(n_lst))
  print "\tInsecure values (MEDIAN): {}, (MEAN): {}, (COUNT): {}".format(np.median(list(i_lst)), np.mean(list(i_lst)), len(i_lst))
  try:
    TS, p = stats.mannwhitneyu(list(i_lst), list(n_lst), alternative="greater")
  except ValueError:
    TS, p = 0.0, 1.0
  cd = cliffs_delta(list(i_lst), list(n_lst), diff)
  print "\tTest Statistic: {}, P-Value: {}, Cliff\'s Effect Size: {}".format(TS, p, cd)

#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#
