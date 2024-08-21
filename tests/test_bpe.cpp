#include <gtest/gtest.h>
#include "../include/BPE.h"

// Fixture for BPE tests
class BPETest : public ::testing::Test {
protected:
    // BPE object to be used in tests
    BPE* bpe{};

    // Set up the BPE object before each test
    void SetUp() override {
        std::map<std::pair<std::string, std::string>, int> merges = {
            {{"l", "o"}, 1},
            {{"lo", "w"}, 2},
            {{"w", "e"}, 3},
            {{"e", "r"}, 4},
            {{"r", "t"}, 5}
        };

        std::unordered_map<std::string, int> vocab = {
            {"low", 0},
            {"er", 1},
            {"lo", 2},
            {"w", 3},
            {"e", 4},
            {"r", 5},
            {"t", 6},
            {"l", 7},
            {"ow", 8},
        };

        bpe = new BPE(merges, vocab);
    }

    // Clean up the BPE object after each test
    void TearDown() override {
        delete bpe;
    }
};

// Test case for the word "lowert"
TEST_F(BPETest, TokenizeLowert) {
    std::vector<std::string> expected = {"low", "er", "t"};
    std::vector<std::string> result = bpe->apply("lowert");
    EXPECT_EQ(result, expected);
}

// Test case for the word "lower"
TEST_F(BPETest, TokenizeLower) {
    std::vector<std::string> expected = {"low", "er"};
    std::vector<std::string> result = bpe->apply("lower");
    EXPECT_EQ(result, expected);
}

// Test case for the word "slowest"
TEST_F(BPETest, TokenizeSlowest) {
    std::vector<std::string> expected = {"s", "low", "e", "s", "t"};
    std::vector<std::string> result = bpe->apply("slowest");
    EXPECT_EQ(result, expected);
}