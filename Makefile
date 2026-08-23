export CXX=clang++
export LIBCRACKLE=/Users/$(USER)/code/crackle
export CRACKLE_INCLUDES=-I$(LIBCRACKLE)/src/ -I$(LIBCRACKLE)/third_party/fastcrc/

test: gaara_test.cpp
	./automated_tests

profile:
	$(CXX) -g -Og -std=c++20 -Isrc $(CRACKLE_INCLUDES) test.cpp -o test
	rm -rf test_profile.trace
	xctrace record --template "Time Profiler" --launch --output test_profile.trace --time-limit 5m -- ./test
	open test_profile.trace

expt:
	$(CXX) -g -O3 -std=c++20 -Isrc $(CRACKLE_INCLUDES) test.cpp -o test

gaara_test.cpp:
	$(CXX) -Og -g src/gaara_test.cpp $(shell pkg-config --cflags --libs gtest gtest_main) -std=c++20 -o automated_tests

python:
	rm -rf build
	pip install -e . --no-build-isolation