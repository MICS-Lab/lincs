#####################
# Top-level targets #
#####################

.PHONY: default
default: test

#############
# Inventory #
#############

all_source_files=$(shell find src -name '*.cu')
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
	@nvcc -c $< -o $@
