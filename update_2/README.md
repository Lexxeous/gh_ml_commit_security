# Update 2

### Author(s)
Jonathan A. Gibson, Andrew P. Worley, Raunak Shakya

### Objectives

Type-based model: REPO_TYPE
Size-based model: ADD_LOC, DEL_LOC, TOT_LOC
Time-based model: PRIOR_AGE
Full model: ADD_LOC, DEL_LOC, TOT_LOC, PRIOR_AGE, REPO_TYPE
<br>
Repeat the following steps for type-based model, size-based model, time-based model, and full model:

1. Take CSV as input, separate out independent columns(s), and the dependent column is SECU_FLAG.
2. Apply Decision Tree, Random Forest, ANN (Artificial Neural Network), kNN (k Nearest Neighbor), & Naïve Bayes classification algorithms.
3. Apply 10-by-10 fold cross validation, and then report prediction accuracy using precision, recall, and F-measure.

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
