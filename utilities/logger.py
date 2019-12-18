#!/usr/local/bin/python3

import sys

"""
Filename: update_1.py
Authors: Amith Koujalgi, Eric Leschinski, & Jonathan A. Gibson
Description:
  The "Logger" class will print the Python script output to a specified log file
  as well as print the output to the screen just by using the default "print" keyword.
Notes:
  Change the "sys.argv[1]" variable to a different command line argument
  or a hardcoded log file to fit your application.
"""
class Logger(object):
  def __init__(self):
    self.terminal = sys.stdout
    self.log = open(sys.argv[1], "a") # will open the log file and append session runs

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    """
    The flush method is needed for Python3 compatibility.
    This handles the flush command by doing nothing.
    You might want to specify some extra behavior here.
    """
    pass    

sys.stdout = Logger()