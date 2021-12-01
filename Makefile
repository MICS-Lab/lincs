# Copyright 2021 Vincent Jacques

############################
# Default top-level target #
############################

.PHONY: default
default: dep-graph lint test tools

#############
# Inventory #
#############

# Source files
tools_source_files := $(wildcard tools/*.cpp)
header_files := $(wildcard library/*.hpp)
cpp_lib_source_files := $(wildcard library/*.cpp)
cu_lib_source_files := $(wildcard library/*.cu)
cpp_test_source_files := $(wildcard */*-tests.cpp)
cu_test_source_files := $(wildcard */*-tests.cu)
sh_test_source_files := $(wildcard */*-tests.sh)

# Intermediate files
object_files := $(patsubst %.cpp,build/obj/%.o,$(cpp_lib_source_files) $(tools_source_files)) $(patsubst %.cu,build/obj/%.o,$(cu_lib_source_files))
dependency_includes := $(patsubst %.cpp,build/deps/%.deps,$(cpp_lib_source_files) $(tools_source_files)) $(patsubst %.cu,build/deps/%.deps,$(cu_lib_source_files))
non_compilation_includes := $(patsubst %-tests.cpp,build/tests/%-non-compilation-tests.deps, $(cpp_test_source_files)) $(patsubst %-tests.cu,build/tests/%-non-compilation-tests.deps, $(cu_test_source_files))

# Sentinel files
cpplint_sentinel_files := $(patsubst %,build/lint/%.cpplint.ok,$(tools_source_files) $(header_files) $(cpp_lib_source_files) $(cu_lib_source_files))
test_sentinel_files := $(patsubst %,build/tests/%.ok,$(cpp_test_source_files) $(cu_test_source_files) $(sh_test_source_files))

# Final products
tools_binary_files := $(patsubst tools/%.cpp,build/tools/bin/%,$(tools_source_files))

.PHONY: debug-inventory
debug-inventory:
	@echo "tools_source_files:\n$(tools_source_files)\n"
	@echo "header_files:\n$(header_files)\n"
	@echo "cpp_lib_source_files:\n$(cpp_lib_source_files)\n"
	@echo "cu_lib_source_files:\n$(cu_lib_source_files)\n"
	@echo "cpp_test_source_files:\n$(cpp_test_source_files)\n"
	@echo "cu_test_source_files:\n$(cu_test_source_files)\n"
	@echo "sh_test_source_files:\n$(sh_test_source_files)\n"
	@echo "object_files:\n$(object_files)\n"
	@echo "dependency_includes:\n$(dependency_includes)\n"
	@echo "non_compilation_includes:\n$(non_compilation_includes)\n"
	@echo "cpplint_sentinel_files:\n$(cpplint_sentinel_files)\n"
	@echo "test_sentinel_files:\n$(test_sentinel_files)\n"
	@echo "tools_binary_files:\n$(tools_binary_files)"

###############################
# Secondary top-level targets #
###############################

.PHONY: tools
tools: $(tools_binary_files)

.PHONY: dep-graph
dep-graph: build/dependency-graph.png

.PHONY: compile
compile: $(object_files)

##########################
# Automated dependencies #
##########################

$(foreach file,$(dependency_includes),$(eval include $(file)))

build/deps/%.deps: %.cu builder/fix-g++-MM.py
	@echo "nvcc -MM $< -o $@"
	@mkdir -p $(dir $@)
	@g++ -MM -x c++ $< | python3 builder/fix-g++-MM.py build/obj/$*.o $@ >$@

build/deps/%.deps: %.cpp builder/fix-g++-MM.py
	@echo "g++  -MM $< -o $@"
	@mkdir -p $(dir $@)
	@g++ -MM -x c++ $< | python3 builder/fix-g++-MM.py build/obj/$*.o $@ >$@

build/dependency-graph.png: builder/deps-to-dot.py $(dependency_includes)
	@echo "cat *.deps | dot -o $@"
	@mkdir -p $(dir $@)
	@cat $(dependency_includes) | python3 builder/deps-to-dot.py | tred | dot -Tpng -o $@

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

build/tests/library/initialize-tests: \
  build/obj/library/initialize.o \
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

build/tests/library/learning-optim-tests.sh.ok: \
  build/tools/bin/generate-learning-set \
  build/tools/bin/generate-model \
  build/tools/bin/learn

build/tests/tools/generate-learning-set-tests.sh.ok: \
  build/tools/bin/generate-learning-set \
  build/tools/bin/generate-model

build/tests/tools/generate-model-tests.sh.ok: \
  build/tools/bin/generate-model

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

.PHONY: test
test: $(test_sentinel_files)

# Unit-ish tests

# Run once without Valgrind to check multi-threaded behavior,
# then once with Valgrind to check for invalid memory accesses.
# (Valgrind effectively serializes all threads)

build/tests/%-tests.cu.ok: build/tests/%-tests
	@echo "$<"
	@mkdir -p $(dir $@)
	@$<
	@timeout 300 valgrind --exit-on-first-error=yes --error-exitcode=1 $<
	@touch $@

build/tests/%-tests.cpp.ok: build/tests/%-tests
	@echo "$<"
	@mkdir -p $(dir $@)
	@$<
	@timeout 300 valgrind --exit-on-first-error=yes --error-exitcode=1 $<
	@touch $@

# Non-compilation tests

$(foreach file,$(non_compilation_includes),$(eval include $(file)))

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
	@cd $@-wd && BUILD_DIR=../../.. timeout 300 bash ../../../../$<
	@touch $@

########
# Link #
########

# Of test executables

build/tests/%-tests: build/obj/%-tests.o
	@echo "nvcc    $< -o $@"
	@mkdir -p $(dir $@)
	@nvcc $^ -lgtest_main -lgtest -lortools -Xcompiler -fopenmp -o $@

# Of tools

build/tools/bin/%: build/obj/tools/%.o
	@echo "nvcc    $< -o $@"
	@mkdir -p $(dir $@)
	@nvcc $^ -lortools -Xcompiler -fopenmp -o $@

###############
# Compilation #
###############

NVCC_COMPILE_OPTIONS=-dc -std=c++17 -g --expt-relaxed-constexpr -Xcompiler -Wall,-Wextra,-Werror,-fopenmp
GPP_COMPILE_OPTIONS=-std=c++17 -g -c -I/usr/local/cuda-11.2/targets/x86_64-linux/include -Wall -Wextra -Wpedantic -Werror -fopenmp

build/obj/%.o: %.cu
	@echo "nvcc -c $< -o $@"
	@mkdir -p $(dir $@)
	@nvcc $(NVCC_COMPILE_OPTIONS) $< -o $@

build/obj/%.o: %.cpp
	@echo "g++  -c $< -o $@"
	@mkdir -p $(dir $@)
	@g++ $(GPP_COMPILE_OPTIONS) $< -o $@
