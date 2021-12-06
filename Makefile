# Copyright 2021 Vincent Jacques

############################
# Default top-level target #
############################

.PHONY: default
default: dep-graph lint test tools

#############
# Inventory #
#############

source_directories=library tools

all_source_files=$(shell find $(source_directories) -name '*.cu' -or -name '*.cpp')
all_header_files=$(shell find $(source_directories) -name '*.hpp')
test_source_files=$(shell find $(source_directories) -name '*-tests.cu' -or -name '*-tests.cpp')
test_shell_files=$(shell find $(source_directories) -name '*-tests.sh')
tools_source_files=$(shell find tools -name '*.cu' -or -name '*.cpp')

tools=$(foreach file,$(tools_source_files),$(patsubst tools/%.cpp,build/tools/bin/%,$(patsubst tools/%.cu,build/tools/bin/%,$(file))))

###############################
# Secondary top-level targets #
###############################

.PHONY: tools
tools: $(tools)

.PHONY: dep-graph
dep-graph: build/dependency-graph.png

.PHONY: compile
compile: $(foreach file,$(all_source_files),$(patsubst %.cpp,build/obj/%.o,$(patsubst %.cu,build/obj/%.o,$(file))))

##########################
# Automated dependencies #
##########################

dependency_files=$(foreach file,$(all_source_files),$(patsubst %.cpp,build/deps/%.deps,$(patsubst %.cu,build/deps/%.deps,$(file))))

$(foreach file,$(dependency_files),$(eval include $(file)))

build/deps/%.deps: %.cu builder/fix-g++-MM.py
	@echo "nvcc -MM $< -o $@"
	@mkdir -p $(dir $@)
	@g++ -MM -x c++ $< | python3 builder/fix-g++-MM.py build/obj/$*.o $@ >$@

build/deps/%.deps: %.cpp builder/fix-g++-MM.py
	@echo "g++  -MM $< -o $@"
	@mkdir -p $(dir $@)
	@g++ -MM -x c++ $< | python3 builder/fix-g++-MM.py build/obj/$*.o $@ >$@

build/dependency-graph.png: builder/deps-to-dot.py $(dependency_files)
	@echo "cat *.deps | dot -o $@"
	@mkdir -p $(dir $@)
	@cat $(dependency_files) | python3 builder/deps-to-dot.py | tred | dot -Tpng -o $@

#######################
# Manual dependencies #
#######################

# Of test executables to object files

build/tests/library/assign-tests: \
  build/obj/library/assign.o \
  build/obj/library/generate.o \
  build/obj/library/io.o \
  build/obj/library/problem.o \
  build/obj/library/stopwatch.o \
  build/obj/library/test-utils.o

build/tests/library/problem-tests: \
  build/obj/library/io.o \
  build/obj/library/problem.o \
  build/obj/library/test-utils.o

build/tests/library/improve-profiles-tests: \
  build/obj/library/assign.o \
  build/obj/library/generate.o \
  build/obj/library/improve-profiles.o \
  build/obj/library/io.o \
  build/obj/library/problem.o \
  build/obj/library/randomness.o \
  build/obj/library/stopwatch.o \
  build/obj/library/test-utils.o

build/tests/library/improve-weights-tests: \
  build/obj/library/assign.o \
  build/obj/library/generate.o \
  build/obj/library/improve-weights.o \
  build/obj/library/io.o \
  build/obj/library/problem.o \
  build/obj/library/stopwatch.o \
  build/obj/library/test-utils.o

build/tests/library/io-tests: \
  build/obj/library/io.o

build/tests/library/learning-tests: \
  build/obj/library/assign.o \
  build/obj/library/generate.o \
  build/obj/library/improve-profiles.o \
  build/obj/library/improve-weights.o \
  build/obj/library/initialize.o \
  build/obj/library/io.o \
  build/obj/library/learning.o \
  build/obj/library/problem.o \
  build/obj/library/randomness.o \
  build/obj/library/stopwatch.o

build/tests/library/randomness-tests: \
  build/obj/library/randomness.o

# Of test shell scripts to tools

build/tests/library/learning-tests.sh.ok: \
  build/tools/bin/generate-learning-set \
  build/tools/bin/generate-model \
  build/tools/bin/learn

build/tests/tools/learn-tests.sh.ok: \
  build/tools/bin/generate-learning-set \
  build/tools/bin/generate-model \
  build/tools/bin/learn

# Of tools to object files

build/tools/bin/generate-model: \
  build/obj/library/assign.o \
  build/obj/library/generate.o \
  build/obj/library/io.o \
  build/obj/library/problem.o \
  build/obj/library/randomness.o \
  build/obj/library/stopwatch.o

build/tools/bin/generate-learning-set: \
  build/obj/library/assign.o \
  build/obj/library/generate.o \
  build/obj/library/io.o \
  build/obj/library/problem.o \
  build/obj/library/randomness.o \
  build/obj/library/stopwatch.o

build/tools/bin/learn: \
  build/obj/library/assign.o \
  build/obj/library/improve-profiles.o \
  build/obj/library/improve-weights.o \
  build/obj/library/initialize.o \
  build/obj/library/io.o \
  build/obj/library/learning.o \
  build/obj/library/problem.o \
  build/obj/library/randomness.o \
  build/obj/library/stopwatch.o

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

# Unit-ish tests

build/tests/%-tests.cu.ok: build/tests/%-tests
	@echo "$<"
	@mkdir -p $(dir $@)
	@valgrind --exit-on-first-error=yes --error-exitcode=1 $<
	@touch $@

build/tests/%-tests.cpp.ok: build/tests/%-tests
	@echo "$<"
	@mkdir -p $(dir $@)
	@valgrind --exit-on-first-error=yes --error-exitcode=1 $<
	@touch $@

# Non-compilation tests

$(foreach file,$(test_source_files),$(eval include $(patsubst %-tests.cpp,build/tests/%-non-compilation-tests.deps,$(patsubst %-tests.cu,build/tests/%-non-compilation-tests.deps,$(file)))))

build/tests/%-non-compilation-tests.deps: builder/make-non-compilation-tests-deps.py %-tests.cu
	@echo $^
	@mkdir -p $(dir $@)
	@python3 $^ >$@

build/tests/%-non-compilation-tests.deps: builder/make-non-compilation-tests-deps.py %-tests.cpp
	@echo $^
	@mkdir -p $(dir $@)
	@python3 $^ >$@

# Integration-ish tests

build/tests/%-tests.sh.ok: %-tests.sh
	@echo "$<"
	@mkdir -p $(dir $@)
	@rm -rf $@-wd
	@mkdir -p $@-wd
	@cd $@-wd && BUILD_DIR=../../.. bash ../../../../$<
	@touch $@

########
# Link #
########

# Of test executables

build/tests/%-tests: build/obj/%-tests.o
	@echo "nvcc    $< -o $@"
	@mkdir -p $(dir $@)
	@nvcc $^ -lgtest_main -lgtest -lortools -o $@

# Of tools

build/tools/bin/%: build/obj/tools/%.o
	@echo "nvcc    $< -o $@"
	@mkdir -p $(dir $@)
	@nvcc $^ -lortools -o $@

###############
# Compilation #
###############

NVCC_COMPILE_OPTIONS=-dc -std=c++17 -g --expt-relaxed-constexpr -Xcompiler -Wall,-Wextra,-Werror
GPP_COMPILE_OPTIONS=-std=c++17 -g -c -I/usr/local/cuda-11.2/targets/x86_64-linux/include -Wall -Wextra -Wpedantic -Werror

build/obj/%.o: %.cu
	@echo "nvcc -c $< -o $@"
	@mkdir -p $(dir $@)
	@nvcc $(NVCC_COMPILE_OPTIONS) $< -o $@

build/obj/%.o: %.cpp
	@echo "g++  -c $< -o $@"
	@mkdir -p $(dir $@)
	@g++ $(GPP_COMPILE_OPTIONS) $< -o $@
