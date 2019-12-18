# Update 3

### Author(s)
Jonathan A. Gibson, Andrew P. Worley, Raunak Shakya

### Objectives
The objectives for this update build on the objectives from *Update 2* while applying transfer learning and resampling with the prior classification models and a Deep Neural Network. Results are compared before and after resampling.

1. Identify and partition dataset groups: "ComputationalChemistry", "Astronomy", "ComputationalBiology", and "Others". "OTHERS" will include "ComputationalScience" & "ComputationalIntelligence".
2. For each dataset (partitioned dataframe), use Mann Whitney U Tests and Cliff's Effect Size to find differences between "INSECURE" & "NEUTRAL" "SECU_FLAG" labels.
3. Apply full model: (ADD_LOC, DEL_LOC, TOT_LOC,DEV_EXP, DEV_RECENT, PRIOR_AGE, CHANGE_FILE_CNT). Build model for each type of dataset, then use one dataset to train and test against others. Represent the following 12 (Train, Test) pairs for each of the 5 classification models (60 result sets).
4. Apply deep neural network to compare against classification algorithms, for each pair.
5. Apply SMOTE on training datasets for each pair, and then build models with all learners.
6. Report confusion matrix, precision, recall, and F-measure for Item 2, Item 3, and Item 4. Report on Items 3 & 4 should be a "before and after" result based on the lack and use of SMOTE sampling respectively.

The 12 pairs for transfer learning will follow the format:
```txt

[
[(Chemistry, Astronomy), (Chemistry ,Biology), (Chemistry, Others)],
[(Astronomy, Biology), (Astronomy, Chemistry), (Astronomy, Others)],
[(Biology, Astronomy), (Biology, Chemistry), (Biology, Others)],
[(Others, Biology), (Others, Astronomy), (Others, Chemistry)]
]
```

### Disclaimer

You must install **Keras**, **Tensorflow**, **TQDM**, and **Imbalanced-Learn** using `pip install <pkg_name>` for **Python2** or `pip3 install <pkg_name>` for **Python3** for this program to work properly.

The latest version of `imbalanced-learn` that works with Python2 is `v0.4.2`. If you want to run the program with a **Python2** environment, you must install the proper version. If you want to run the program with a **Python3** environment, you can run it with the latest version, as of `v0.5.0`.

### Instructions

You can run the program with **Python2** by using `make run2` or with **Python3** using `make run3`. If you run into an issue with running the program locally with **Python3**, you may be able to solve the problem in two different ways.

#### Fix 1
First, you could create a virtual environment by using the following steps:
```sh
$ pip3 install virtualenv # install "virtualenv"
$ virtualenv <virtual_proj_name> # create a virtual project with name: <virtual_proj_name>
$ cd <virtual_proj_name>/bin # navigate to the virtual project's bin folder
$ chmod 777 activate # gives rights to run the "activate" executable
$ source ./activate # activate the virtual environment
$ cd ../.. # you can navigate through folders with this virtual sandbox back to your project folder
```
Once inside the virtual environment, you can install any **Python** libraries that you wish using `pip3 install <libname>` and isolate the environment from other **Python** projects. When you are ready to exit the virtual environment, just use `deactivate`.

#### Fix 2 (recommended)
Second, you may have **Python3** installed incorrectly (like I did). Although macOS machines have **Python2** pre-installed, we can also setup a third-party installation. These instructions are intended for a Unix environment.

*Visit*: https://stackoverflow.com/questions/3819449/how-to-uninstall-python-2-7-on-a-mac-os-x-10-6-4/3819829#3819829 to properly uninstall any desired version of **Python**.
<br>
*Visit*: https://docs.python-guide.org/starting/install3/osx/ to properly install the Homebrew versions of **Python2** and **Python3**.

### Notes
SKLearn Naïve Bayes Classification Function: <br> https://scikit-learn.org/stable/modules/naive_bayes.html <br><br>
SKLearn K-Nearest Neighbor Classification Function: <br> https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html <br><br>
SKLearn Decision Tree Classification Function: <br> https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html <br><br>
Using Decision Trees with Pandas: <br> http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html <br><br>
SKLearn Artificial Neural Network Classification Function: <br> https://scikit-learn.org/stable/modules/neural_networks_supervised.html <br><br>
SKLearn Random Forest Classification Function: <br> https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html <br><br>
SKLearn Cross Validation Function: <br> https://scikit-learn.org/stable/modules/cross_validation.html <br><br>
SKLearn Cross Validate Function: <br> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html <br><br>
SKLearn Cross Validate Score Function: <br> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html <br><br>
SKLearn Metrics F1-Score Function: <br> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html <br><br>
DeepAI Glossary F1-Score: <br> https://deepai.org/machine-learning-glossary-and-terms/f-score <br><br>
DeepAI Glossary Precision and Recall: <br> https://deepai.org/machine-learning-glossary-and-terms/precision-and-recall <br><br>
Imbalanced Learn Documentation: <br> https://imbalanced-learn.readthedocs.io/en/stable/api.html <br><br>
Using SMOTE with Imbalanced Data in Python: <br> https://www.kaggle.com/qianchao/smote-with-imbalance-data <br><br>
Proposal Paper for SMOTEENN and SMOTETomek: <br> http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.7757&rep=rep1&type=pdf <br><br>
Keras Documentation: <br> https://keras.io <br><br>
Building Your First DNN with Keras: <br> https://machinelearningmastery.com/tutorial-first-neural-network-python-keras <br><br>
Connect DNN with Classification Model: <br> https://datascience.stackexchange.com/questions/51100/keras-how-to-connect-a-cnn-model-with-a-decision-tree <br><br>
Return More Metrics for Keras DNN Model: <br> https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model <br><br>

#### ENN (Edited Nearest Neighbors) Cleaning and Tomek Links Cleaning
Cleaning techniques remove the decision boundary overlapping that is introduced from sampling methods. Majority class examples might be invading the minority class space and vice-versa. ENN tends to remove more examples than the Tomek links does, so it is expected that it will provide a more in depth data cleaning. Any example that is misclassified by ENN's three nearest neighbors is removed from the training set.



