# Copyright 2021 Vincent Jacques

#####################
# Top-level targets #
#####################

.PHONY: default
default: lint test tools

#############
# Inventory #
#############

source_directories=library tools

all_source_files=$(shell find $(source_directories) -name '*.cu')
all_header_files=$(shell find $(source_directories) -name '*.hpp')
test_source_files=$(shell find $(source_directories) -name '*-tests.cu')
test_shell_files=$(shell find $(source_directories) -name '*-tests.sh')
tools_source_files=$(shell find tools -name '*.cu')

tools=$(foreach file,$(tools_source_files),$(patsubst tools/%.cu,build/tools/bin/%,$(file)))

.PHONY: tools
tools: $(tools)

##########################
# Automated dependencies #
##########################

$(foreach file,$(foreach file,$(all_source_files),$(patsubst %.cu,build/deps/%.deps,$(file))),$(eval include $(file)))

build/deps/%.deps: %.cu builder/fix-g++-MM.py
	@echo "nvcc -MM $< -o $@"
	@mkdir -p $(dir $@)
	@g++ -MM -x c++ $< | python3 builder/fix-g++-MM.py build/obj/$*.o $@ >$@

#######################
# Manual dependencies #
#######################

build/tools/bin/generate-model: build/obj/library/generate.o build/obj/library/io.o build/obj/library/improve-profiles.o
build/tools/bin/generate-learning-set: build/obj/library/generate.o build/obj/library/io.o build/obj/library/improve-profiles.o
build/tests/library/improve-profiles-tests: build/obj/library/io.o
build/tools/bin/test-improve-profiles: build/obj/library/io.o build/obj/library/improve-profiles.o

########
# Lint #
########

cpplint_sentinel_files=$(foreach file,$(all_source_files) $(all_header_files),$(patsubst %,build/lint/%.cpplint.ok,$(file)))

.PHONY: lint
lint: $(cpplint_sentinel_files)

build/lint/%.cpplint.ok: %
	@echo "cpplint $<"
	@mkdir -p $(dir $@)
	@cpplint --root=library --linelength=120 $<
	@touch $@

#########
# Tests #
#########

test_sentinel_files=$(foreach file,$(test_source_files) $(test_shell_files),$(patsubst %,build/tests/%.ok,$(file)))

.PHONY: test
test: $(test_sentinel_files)

# - unit-ish tests
build/tests/%-tests.cu.ok: build/tests/%-tests
	@echo "$<"
	@mkdir -p $(dir $@)
	@./$<
	@touch $@

# - non-compilation tests

$(foreach file,$(foreach file,$(test_source_files),$(patsubst %-tests.cu,build/tests/%-non-compilation-tests.deps,$(file))),$(eval include $(file)))

build/tests/%-non-compilation-tests.deps: builder/make-non-compilation-tests-deps.py %-tests.cu
	@echo $^
	@mkdir -p $(dir $@)
	@python3 $^ >$@

# - integration-ish tests
build/tests/%-tests.sh.ok: %-tests.sh $(tools)
	@echo "$<"
	@mkdir -p $(dir $@)
	@rm -rf $@-wd
	@mkdir -p $@-wd
	@cd $@-wd && BUILD_DIR=../../.. bash ../../../../$<
	@touch $@

########
# Link #
########

# - of test executables
build/tests/%-tests: build/obj/%-tests.o build/obj/%.o
	@echo "nvcc    $< -o $@"
	@mkdir -p $(dir $@)
	@nvcc $^ -lgtest_main -lgtest -o $@

# - of tools
build/tools/bin/%: build/obj/tools/%.o
	@echo "nvcc    $< -o $@"
	@mkdir -p $(dir $@)
	@nvcc $^ -o $@

###############
# Compilation #
###############

build/obj/%.o: %.cu
	@echo "nvcc -c $< -o $@"
	@mkdir -p $(dir $@)
	@nvcc -std=c++17 -g --expt-relaxed-constexpr -c $< -o $@
