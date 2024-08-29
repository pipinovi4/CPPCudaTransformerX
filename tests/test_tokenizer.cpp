#include "gtest/gtest.h"
#include "../include/Tokenizer.h"

class TokenizerTest : public ::testing::Test {
protected:
    Tokenizer<std::string>* tokenizer;
    std::vector<std::string> sample_text;

    TokenizerTest() : tokenizer(nullptr) {}

    void SetUp() override {
        sample_text = {"This", "is", "a", "sample", "sentence"};
        tokenizer = new Tokenizer<std::string>(10, " ", true, true, true, true);
    }

    void TearDown() override {
        delete tokenizer;
    }
};

TEST_F(TokenizerTest, HandlesVocabularyBuilding)
{
    const std::vector<std::string> dataset = {"This", "is", "a", "sample", "sentence"};

    const auto vocab = Tokenizer<std::string>::buildVocabulary(dataset);
    tokenizer->setVocabulary(vocab);

    EXPECT_EQ(vocab.size(), 9);  // Including 4 special tokens <PAD>, <UNK>, <SOS>, <EOS>
    EXPECT_EQ(tokenizer->textToIds({"this"})[0], 4);
    EXPECT_EQ(tokenizer->textToIds({"sample"})[0], 7);
}

TEST_F(TokenizerTest, HandlesTextToIds) {
    const std::vector<std::string> dataset = {"This", "is", "a", "sample", "sentence"};
    const auto vocab = Tokenizer<std::string>::buildVocabulary(dataset);
    tokenizer->setVocabulary(vocab);

    const auto ids = tokenizer->textToIds(sample_text);

    EXPECT_EQ(ids.size(), 10);
    EXPECT_EQ(tokenizer->idsToText({4})[0], "this");
    EXPECT_EQ(tokenizer->idsToText({7})[0], "sample");
}

TEST_F(TokenizerTest, HandlesPadding) {
    tokenizer = new Tokenizer<std::string>(10, " ", true, false, true, true); // Enable padding
    const std::vector<std::string> dataset = {"This", "is", "a", "sample", "sentence"};
    const auto vocab = Tokenizer<std::string>::buildVocabulary(dataset);
    tokenizer->setVocabulary(vocab);

    const auto ids = tokenizer->textToIds(sample_text);

    EXPECT_EQ(ids.size(), 10);
    EXPECT_EQ(ids[6], tokenizer->textToIds({"<pad>"})[0]);  // Assuming <PAD> is at index 0
}

TEST_F(TokenizerTest, HandlesTruncation) {
    tokenizer = new Tokenizer<std::string>(3, " ", false, true, true, true); // Enable truncation
    const std::vector<std::string> dataset = {"This", "is", "a", "sample", "sentence"};
    const auto vocab = Tokenizer<std::string>::buildVocabulary(dataset);
    tokenizer->setVocabulary(vocab);

    const auto ids = tokenizer->textToIds(sample_text);

    EXPECT_EQ(ids.size(), 3);
    EXPECT_EQ(ids[0], tokenizer->textToIds({"this"})[0]);
    EXPECT_EQ(ids[2], tokenizer->textToIds({"a"})[0]);  // Only the first 3 tokens should remain
}

TEST_F(TokenizerTest, HandlesSpecialTokens) {
    const std::vector<std::string> dataset = {"This", "is", "a", "sample", "sentence"};
    auto vocab = Tokenizer<std::string>::buildVocabulary(dataset);
    vocab["<pad>"] = 0;  // Assuming <PAD> is at index 0
    vocab["<unk>"] = 99; // Assuming <UNK> is at index 99
    tokenizer->setVocabulary(vocab);

    const auto ids = tokenizer->textToIds({"This", "is", "unknownword"});

    EXPECT_EQ(ids.size(), 10);
    EXPECT_EQ(ids[2], 99);  // 'unknownword' should map to <UNK>
}
