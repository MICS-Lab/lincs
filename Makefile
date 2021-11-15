# Copyright 2021 Vincent Jacques

#####################
# Top-level targets #
#####################

.PHONY: default
default: lint test

#############
# Inventory #
#############

all_source_files=$(shell find src -name '*.cu')
all_header_files=$(shell find src -name '*.hpp')
test_source_files=$(shell find src  -name '*-tests.cu')

##########################
# Automated dependencies #
##########################

dependency_files=$(foreach file,$(all_source_files),$(patsubst src/%.cu,build/deps/%.deps,$(file)))

$(foreach file,$(dependency_files),$(eval include $(file)))

build/deps/%.deps: src/%.cu
	@echo "nvcc -MM $< -o $@"
	@mkdir -p $(dir $@)
	@g++ -MM -x c++ $< | python3 tools/fix-g++-MM.py build/obj/$*.o $@ >$@

########
# Lint #
########

cpplint_sentinel_files=$(foreach file,$(all_source_files) $(all_header_files),$(patsubst src/%,build/lint/%.cpplint.ok,$(file)))

.PHONY: lint
lint: $(cpplint_sentinel_files)

build/lint/%.cpplint.ok: src/%
	@echo "cpplint $<"
	@mkdir -p $(dir $@)
	@cpplint --root=src --linelength=120 $<
	@touch $@

#########
# Tests #
#########

test_sentinel_files=$(foreach file,$(test_source_files),$(patsubst src/%.cu,build/tests/%.ok,$(file)))

.PHONY: test
test: $(test_sentinel_files)

build/tests/%-tests.ok: build/tests/%-tests
	@echo "$<"
	@mkdir -p $(dir $@)
	@./$<
	@touch $@

########
# Link #
########

# - of test executables
build/tests/%-tests: build/obj/%-tests.o build/obj/%.o
	@echo "nvcc    $^ -o $@"
	@mkdir -p $(dir $@)
	@nvcc $^ -lgtest_main -lgtest -o $@

###############
# Compilation #
###############

build/obj/%.o: src/%.cu
	@echo "nvcc -c $< -o $@"
	@mkdir -p $(dir $@)
	@nvcc -std=c++17 -c $< -o $@
