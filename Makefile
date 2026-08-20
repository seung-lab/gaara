export CXX=clang++

test: gaara_test.cpp
	./automated_tests

profile: test.cpp
	rm -rf test_profile.trace
	xctrace record --template "Time Profiler" --launch --output test_profile.trace --time-limit 5m -- ./test
	open test_profile.trace

test.cpp:
	$(CXX) -g -O3 -std=c++20 -Isrc -Ilibcrackle/ test.cpp -o test

gaara_test.cpp:
	$(CXX) -Og -g src/gaara_test.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lgtest -lgtest_main -std=c++17 -o automated_tests

python:
	rm -rf build
	pip install -e . --no-build-isolation