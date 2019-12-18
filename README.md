# GitHub Machine Learning Commit Security Project

## Summary
The purpose of this project was to write **Python** scripts to automatically analyze GitHub commit data and apply algorithms such as *TF-IDF*, *Mann Whitney-U Tests*, *Cliff's Delta Effect Size*, *Decision Tree Classification* (DTC), *Random Forest Classification* (RFC), *k-Nearest Neighbor Classification* (kNN), *Artificial Neural Network Classification* (ANN), *Naïve Bayes Classification* (NBC), *Deep Neural Network Classification* (DNN), & *SMOTE Resampling* to extract useful information and determine which models work well for the given datasets.

## Documents
The `Documents` directory contains reports about our findings for each of the 3 project updates.

## Updates
Each of the 3 updates contained within this project have different objectives. The author(s), objectives, disclaimers, instructions, & notes relative to each update can be found in thier respective directories, namely: `update_1`, `update_2`, & `update_3`.
<br><br>
Each update directory contains 6 items by default (with an exception for `update_1`): a `Makefile`, the appropriate `README`, a saved output file from a prior successful run of the entire program (`saved_output_u#.out`), the dataset used for the update (`update_#_dataset.csv`), the **Python2** compatible script (`update_#_main_p2.py`), and the **Python3** compatible script (`update_#_main_p3`). The dataset for the first update can be downloaded from my public Google Drive folder (link provided in `update_1/README.md`), as it is very large.
<br><br>
The `Makefile` provided for each update provides quick access to running the **Python2** compatible program with `make run2`, running the **Python3** compatible program with `make run3`, removing the excess log file(s) and compiled **Python** (`.pyc`) update files with `make clean`, removing the excess compiled **Python** (`.pyc`) utilities files with `make clean_utils`, debugging the **Python2** compatible program with `make debug2`, & debugging the **Python3** compatible program with `make debug3`.
<br><br>
Update 3 has a few special dependicies that are laid out in `update_3/README.md`. Additionally, update 3 provides a `skip` boolean variable in the `Setup` section of the script that forces the program to skip over some of the redundant and computationally intensive sections of the script, for faster debugging and testing.

## Utilites
Each update's main scripts have utility wrappers that are referrenced (for both the **Python2** and **Python3** compatible versions, for a total of 6 utility scripts) as well as a custom logger class file that forces the standard output to be appended to a specified log file (`update_#_log.out` by default) while still being outputted to the terminal window.

## Bandit
The last feature available for this project is the ability to run a security analysis on all of the source code using a Python library called `Bandit`. This tool recursively reads through all files and folders within a specified directory, in search of security weaknesses, and reports them with a certain level of severity. 
<br><br>
Test the use of the `Bandit` tool with `make bandit_1` (to scan the first update), `make bandit_2` (to scan the second update), `make bandit_3` (to scan the third update), `make bandit_utils` (to scan the utilities directory), & `make bandit_all` (to scan all updates and the utilities folder). The commands can be used from the root directory of this project. The commands will also exclude 5 particular files and folders from each directory: any potential `__pycache__` folder, and potential virtual environment folder (`/proj` by default), any saved output file (`saved_output_u#.out` by default), any update log file (`update_#_log.out` by default), & the dataset files (`update_#_dataset.csv` by default).
<br><br>
`Bandit` is compatible with **Python2** and **Python3**.
<br><br>
No security weaknesses were found for any of the source code available within this project.