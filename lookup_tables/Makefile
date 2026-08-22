CXX = clang++  # or g++, or any other C++ compiler
CXXFLAGS = -O3

all: simple isthmus
	@mkdir -p bin tables
	@echo "Running programs..."
	./bin/simple
	./bin/isthmus
	xxd -i ./tables/simple.bin > ./tables/simple.hpp
	xxd -i ./tables/isthmus.bin > ./tables/isthmus.hpp

simple: src/simple.cpp
	$(CXX) $(CXXFLAGS) src/simple.cpp -o bin/simple

isthmus: src/isthmus.cpp
	$(CXX) $(CXXFLAGS) src/isthmus.cpp -o bin/isthmus

clean:
	rm bin/simple bin/isthmus
	rm tables/*.bin
