null  :=
space := $(null) #
comma := ,

u1_dir ?= update_1
ex_1_parts := $(u1_dir)/__pycache__ $(u1_dir)/proj $(u1_dir)/saved_output_u1.out $(u1_dir)/update_1_log.out $(u1_dir)/update_1_dataset.csv
ex_1_comp := $(foreach part, $(ex_1_parts), $(part))
ex_1_list := $(subst $(space),$(comma),$(strip $(ex_1_comp)))

u2_dir ?= update_2
ex_2_parts := $(u2_dir)/__pycache__ $(u2_dir)/proj $(u2_dir)/saved_output_u2.out $(u2_dir)/update_2_log.out $(u2_dir)/update_2_dataset.csv
ex_2_comp := $(foreach part, $(ex_2_parts), $(part))
ex_2_list := $(subst $(space),$(comma),$(strip $(ex_2_comp)))

u3_dir ?= update_3
ex_3_parts := $(u3_dir)/__pycache__ $(u3_dir)/proj $(u3_dir)/saved_output_u3.out $(u3_dir)/update_3_log.out $(u3_dir)/update_3_dataset.csv
ex_3_comp := $(foreach part, $(ex_3_parts), $(part))
ex_3_list := $(subst $(space),$(comma),$(strip $(ex_3_comp)))

utils_dir ?= utilities
ex_u_parts := $(utils_dir)/__pycache__
ex_u_comp := $(foreach part, $(ex_u_parts), $(part))
ex_u_list := $(subst $(space),$(comma),$(strip $(ex_u_comp)))

# Run Bandit Security Analysis on Update 1 Directory
bandit_1:
	bandit -r $(u1_dir) -x $(ex_1_list)

# Run Bandit Security Analysis on Update 2 Directory
bandit_2:
	bandit -r $(u2_dir) -x $(ex_2_list)

# Run Bandit Security Analysis on Update 3 Directory
bandit_3:
	bandit -r $(u3_dir) -x $(ex_3_list)

# Run Bandit Security Analysis on Utilities Directory
bandit_utils:
	bandit -r $(utils_dir) -x $(ex_u_list)

# Run Bandit Security Analysis on All Project Folders
bandit_all: bandit_1 bandit_2 bandit_3 bandit_utils