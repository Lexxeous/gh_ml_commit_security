# Fall 2019 - CSC 5220 - Group Project

*Author(s)*: **Jonathan A. Gibson**, **Andrew P. Worley**, **Raunak Shakya**
<br><br>
*Objectives*:

  1. Filter out noisy file names; filenames MUST have extentions and MAY have a path.
  2. Separate dataframes or files into 2 security-related groups: ("INSECURE") and ("NEUTRAL").
  3. Apply text mining (TF-IDF) on the two groups.
  4. Get the text mining matrix, and sort it by TF-IDF scores for both groups.
  5. Take the top 1000 TF-IDF scores for both groups.
  6. Look at the obtained features manually and see what features appear: each member must do it individually then discuss agreements and disagreements.

*Instructions*:
You can run the program with **Python2** by using `make run2` or with **Python3** using `make run3`. There seems to be an issue with running the program locally with **Python3**, without creating a virtual environment. If you run into this issue, follow the following steps:

```sh
$ pip3 install virtualenv # install "virtualenv"
$ virtualenv <virtual_proj_name> # create a virtual project with name: <virtual_proj_name>
$ cd <virtual_proj_name>/bin # navigate to the virtual project's bin folder
$ chmod 777 activate # gives rights to run the "activate" executable
$ source ./activate # activate the virtual environment
$ cd ../.. # you can navigate through folders with this virtual sandbox back to your project folder
```

Once inside the virtual environment, you can install any **Python** libraries that you wish using `pip3 install <libname>` and isolate the environment from other **Python** projects. When you are ready to exit the virtual environment, just use `deactivate`.


*Notes*:
 > For the TF-IDF formula, a token ('t') is a unique modified file name and a script ('s') is a unique commit hash.

*Visit*: https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python for a nice python tutorial on TF–IDF from a list of lists using gensim.
<br>
*See*: https://en.wikipedia.org/wiki/Tf%E2%80%93idf for more information about the TF-IDF text mining algorithm on Wikipedia.
