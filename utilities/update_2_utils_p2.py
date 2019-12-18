# coding: utf-8

#--------------------------------------------- Import Necessary Libraries --------------------------------------------#

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import sklearn.metrics as metric # for printing a classification report
import statistics as stats # for calculating mean, median, and standard deviation
import warnings # for managing warnings
import sys

#------------------------------------------------------- Setup -------------------------------------------------------#

warnings.filterwarnings("ignore") # ignore warnings ; for 0 valued F1 scores

#---------------------------------------------- Utilities Implementation ---------------------------------------------#

''' Add column to dataframe with encoded integers for the classification column target. '''
def encode_target(df, target_column):
  df_mod = df.copy() # modified dataframe
  targets = df_mod[target_column].unique() # NEUTRAL & INSECURE
  map_to_int = {name: n for n, name in enumerate(targets)} # [NEUTRAL, INSECURE] => [0, 1]
  df_mod["TARGET"] = df_mod[target_column].replace(map_to_int)
  print "\nDataset after encoding the target column:", "\n", df_mod.head()
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
  print "\nDataset after encoding the string column:", "\n", df_mod.head()
  return (df_mod, targets)


''' Perform generic classification and calculate cross-validation accuracy, precision, recall, and F1 score. '''
def classification(classifier, features, df, folds, model, targets):
  print "\t", model, "Model Features:", features
  X_data = df[features] # the data to fit
  y_target = df["TARGET"] # the target variable to try to predict
  prediction = cross_val_predict(classifier, X_data, y_target, cv=folds) # return a list of predicted values
  scores = list(cross_val_score(classifier, X_data, y_target, cv=folds)) # return <folds> sized list of cross-validation accuracy values
  scores_rounded = [round(elem, 4) for elem in scores]
  print "\tTesting Phase Accuracy, per fold, across " + str(folds) + " folds:", scores_rounded
  print "\tCross-Validation Score Accuracy: Average %0.4f (± %0.4f) with median %0.4f" % (stats.mean(scores), stats.stdev(scores) * 2, stats.median(scores))
  print "\tCross-Validation Prediction Classification Report:"
  print metric.classification_report(y_target, prediction, digits=4, target_names=targets) # report metrics based on true target values vs. predicted values
  count_true_false_neg_pos(y_target, prediction) # count and print the quantity of TN, FN, FP, & TP
  print "––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––\n" 


''' Count and print the quantity of True Negatives, False Negatives, False Positives, and True Positives '''
def count_true_false_neg_pos(true, pred):
  tn = fn = fp = tp = 0
  for i in range(0, len(pred)):
    if (true[i] == 0 and pred[i] == 0): tn += 1
    elif (true[i] == 0 and pred[i] == 1): fn += 1
    elif (true[i] == 1 and pred[i] == 0): fp += 1
    elif (true[i] == 1 and pred[i] == 1): tp += 1
    else:
      print "ERROR::004::COUNTING_ERROR"
      sys.exit()

  print "\tTrue Negatives: " + str(tn)
  print "\tFalse Negatives: " + str(fn)
  print "\tFalse Positives: " + str(fp)
  print "\tTrue Positives: " + str(tp) + "\n"

#---------------------------------------------------------------------------------------------------------------------#
