CXX = clang++  # or g++, or any other C++ compiler
CXXFLAGS = -O3

all: simple isthmus
	@mkdir -p bin tables
	@echo "Running programs..."
	./bin/simple
	./bin/isthmus

simple: src/simple.cpp
	$(CXX) $(CXXFLAGS) src/simple.cpp -o bin/simple

isthmus: src/isthmus.cpp
	$(CXX) $(CXXFLAGS) src/isthmus.cpp -o bin/isthmus

clean:
	rm bin/simple bin/isthmus
	rm tables/*.bin
