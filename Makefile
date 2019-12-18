null  :=
space := $(null) #
comma := ,

ext := out

cash := __pycache__
virt_env := proj
saved := saved_output_u
pre := update_
log := _log.$(ext)
data := _dataset.csv

u1_dir ?= update_1
ex_1_parts := $(u1_dir)/$(cash) $(u1_dir)/$(virt_env) $(u1_dir)/$(saved)1.$(ext) $(u1_dir)/$(pre)1$(log) $(u1_dir)/$(pre)1$(data)
ex_1_comp := $(foreach part, $(ex_1_parts), $(part))
ex_1_list := $(subst $(space),$(comma),$(strip $(ex_1_comp)))

u2_dir ?= update_2
ex_2_parts := $(u2_dir)/$(cash) $(u2_dir)/$(virt_env) $(u2_dir)/$(saved)2.$(ext) $(u2_dir)/$(pre)2$(log) $(u2_dir)/$(pre)2$(data)
ex_2_comp := $(foreach part, $(ex_2_parts), $(part))
ex_2_list := $(subst $(space),$(comma),$(strip $(ex_2_comp)))

u3_dir ?= update_3
ex_3_parts := $(u3_dir)/$(cash) $(u3_dir)/$(virt_env) $(u3_dir)/$(saved)3.$(ext) $(u3_dir)/$(pre)3$(log) $(u3_dir)/$(pre)3$(data)
ex_3_comp := $(foreach part, $(ex_3_parts), $(part))
ex_3_list := $(subst $(space),$(comma),$(strip $(ex_3_comp)))

utils_dir ?= utilities
ex_u_parts := $(utils_dir)/$(cash)
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