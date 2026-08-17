// brew install google-test
// clang++ -Og -g src/gaara_test.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lgtest -lgtest_main -std=c++17 -o automated_tests && ./automated_tests

#include <gtest/gtest.h>
#include "gaara.hpp"

int add(int a, int b) { return a + b; }

TEST(AddTest, PositiveNumbers) {
    EXPECT_EQ(add(2, 3), 5);
}