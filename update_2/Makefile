log_file ?= "update_2_log.out"
src_pyc2 ?= update_2_main_p2.pyc
src_pyc3 ?= update_2_main_p3.pyc

run2:
	python update_2_main_p2.py $(log_file)

run3:
	python3 update_2_main_p3.py $(log_file)

clean:
	rm $(log_file) $(src_pyc2) $(src_pyc3)

clean_utils:
	cd ../utilities && rm *.pyc

debug:
	python -m pdb update_2_main.py $(log_file)
		# args ('a') –– Gets the argument list of the current function.
		# continue ('c') or ("cont") –– Creates a breakpoint in the current program execution (requires parameters).
		# help ('h') –– Provides a list of commands or help for a specific command.
		# jump ('j') –– Jumps to the next line to be executed.
		# list ('l') –– Prints the source code around the current line.
		# expression ('p') –– Evaluates the expression in the current context and prints the value.
		# pp ('pp') –– "Pretty-prints" the value of the expression.
		# quit ('q') –– Aborts/quits the program.
		# return ('r') –– Continues the program until the current function returns.
