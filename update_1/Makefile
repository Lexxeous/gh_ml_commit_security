log_file ?= "update_1_log.out"
src_pyc2 ?= update_1_main_p2.pyc
src_pyc3 ?= update_1_main_p3.pyc

insecure_hash_files ?= "insecure_hash_files.csv"
insecure_top_features ?= "insecure_top_features.csv"
neutral_hash_files ?= "neutral_hash_files.csv"
neutral_top_features ?= "neutral_top_features.csv"
validated_file_names ?= "validated_file_names.csv"

run2:
	python2 update_1_main_p2.py $(log_file)

run3:
	python3 update_1_main_p3.py $(log_file)

clean:
	rm $(log_file) $(src_pyc_p2) $(src_pyc_p3) $(insecure_hash_files) $(insecure_top_features) $(neutral_hash_files) $(neutral_top_features) $(validated_file_names)

clean_utils:
	cd ../utilities && rm *.pyc

debug_2:
	python -m pdb update_1_main_p2.py $(log_file)

debug_3:
	python -m pdb update_1_main_p3.py $(log_file)


# args ('a') –– Gets the argument list of the current function.
# continue ('c') or ("cont") –– Creates a breakpoint in the current program execution (requires parameters).
# help ('h') –– Provides a list of commands or help for a specific command.
# jump ('j') –– Jumps to the next line to be executed.
# list ('l') –– Prints the source code around the current line.
# expression ('p') –– Evaluates the expression in the current context and prints the value.
# pp ('pp') –– "Pretty-prints" the value of the expression.
# quit ('q') –– Aborts/quits the program.
# return ('r') –– Continues the program until the current function returns.