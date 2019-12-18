# Update 1

### Author(s)
Jonathan A. Gibson, Andrew P. Worley, Raunak Shakya

### Objectives

  1. Filter out noisy file names; filenames MUST have extentions and MAY have a path.
  2. Separate dataframes or files into 2 security-related groups: ("INSECURE") and ("NEUTRAL").
  3. Apply text mining (TF-IDF) on the two groups.
  4. Get the text mining matrix, and sort it by TF-IDF scores for both groups.
  5. Take the top 1000 TF-IDF scores for both groups.
  6. Look at the obtained features manually and see what features appear.

### Instructions
You will need to download the `update_1_dataset.csv` file from my public Google Drive directory. It is approximately 1.1GB in size and will need to be places in the `update_1/` directory of this project. You can download the file here: https://drive.google.com/open?id=1YOchYsdFg_12fMIIlpDJ-BfzNWJ58tAA.

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
 > For the TF-IDF formula, a token ('t') is a unique modified file name and a script ('s') is a unique commit hash.

*Visit*: https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python for a nice python tutorial on TF–IDF from a list of lists using gensim.
<br>
*See*: https://en.wikipedia.org/wiki/Tf%E2%80%93idf for more information about the TF-IDF text mining algorithm on Wikipedia.
