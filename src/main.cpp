#include <algorithm>
#include <cctype>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../include/LossFunction.h"
#include "../include/Optimizer.h"
#include "../models/Transformer.h"
#include "../utils/loadVocab.h"

namespace {

constexpr int kMaxContextTokens = 4;

std::vector<std::string> splitTokens(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string token;

    while (stream >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

std::string joinTokens(const std::vector<std::string>& tokens) {
    std::ostringstream stream;

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) {
            stream << ' ';
        }
        stream << tokens[index];
    }

    return stream.str();
}

std::string toLower(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        }
    );
    return value;
}

bool confirm(const std::string& prompt) {
    std::cout << prompt;

    std::string response;
    if (!std::getline(std::cin, response)) {
        return false;
    }

    response = toLower(response);
    return response == "yes" || response == "y";
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string vocabPath =
        argc > 1 ? argv[1] : "../data/vocab/vocab_30000_words.txt";
    const std::string weightsPath =
        argc > 2 ? argv[2] : "../data/weights/transformer_weights.txt";

    try {
        std::cout << "BackpropLab experimental text-generation demo\n";
        std::cout << "Enter context (1-" << kMaxContextTokens << " tokens): ";

        std::string inputText;
        if (!std::getline(std::cin, inputText)) {
            std::cerr << "Unable to read the input context.\n";
            return 1;
        }

        std::vector<std::string> tokens = splitTokens(inputText);
        if (tokens.empty()) {
            std::cerr << "The context must contain at least one token.\n";
            return 1;
        }

        if (tokens.size() > kMaxContextTokens) {
            tokens.resize(kMaxContextTokens);
            inputText = joinTokens(tokens);

            std::cout << "Context was reduced to " << kMaxContextTokens
                      << " tokens: " << inputText << '\n';

            if (!confirm("Continue? (yes/no): ")) {
                std::cout << "Exiting.\n";
                return 0;
            }
        }

        const int contextTokenCount = static_cast<int>(tokens.size());

        std::cout << "Loading vocabulary from " << vocabPath << "...\n";
        const std::vector<std::string> vocab = load_vocab(vocabPath);
        std::cout << "Vocabulary loaded.\n";

        constexpr int maxTokens = 8;
        constexpr int dModel = 32;
        constexpr int numberOfHeads = 8;
        constexpr int feedForwardSize = 128;
        constexpr float learningRate = 0.001F;
        constexpr float decayRate = 0.9F;
        constexpr float weightDecay = 0.001F;
        constexpr float beta1 = 0.9F;
        constexpr float beta2 = 0.999F;
        constexpr float epsilon = 1e-8F;
        constexpr float labelSmoothing = 0.1F;
        constexpr float dropout = 0.1F;

        const int vocabularySize = static_cast<int>(vocab.size());

        Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule
            learningRateSchedule(learningRate, decayRate);
        LossFunction<float>::crossEntropyLoss lossFunction;
        Optimizer<float>::Adam optimizer(
            learningRate,
            learningRateSchedule,
            weightDecay,
            beta1,
            beta2,
            epsilon
        );

        Transformer<float> transformer(
            &lossFunction,
            &optimizer,
            vocab,
            learningRateSchedule,
            vocabularySize,
            dModel,
            numberOfHeads,
            feedForwardSize,
            maxTokens,
            dropout,
            labelSmoothing
        );

        std::cout << "Loading model weights from " << weightsPath << "...\n";
        transformer.load_weights(weightsPath);

        if (confirm("Show model configuration? (yes/no): ")) {
            std::cout << "\nModel configuration:\n"
                      << "Vocabulary size: " << vocabularySize << '\n'
                      << "Maximum tokens: " << maxTokens << '\n'
                      << "Model dimension: " << dModel << '\n'
                      << "Attention heads: " << numberOfHeads << '\n'
                      << "Feed-forward size: " << feedForwardSize << '\n'
                      << "Learning rate: " << learningRate << '\n'
                      << "Weight decay: " << weightDecay << '\n'
                      << "Label smoothing: " << labelSmoothing << '\n'
                      << "Configured dropout: " << dropout
                      << " (not currently applied)\n";
        }

        const std::vector<std::vector<std::string>> generatedText =
            transformer.generate(
                {transformer.positional_encoder_->tokenize(inputText)},
                contextTokenCount
            );

        if (generatedText.empty()) {
            std::cerr << "The model returned no generated sentences.\n";
            return 1;
        }

        std::cout << "\nGenerated tokens:\n";
        for (const auto& sentence : generatedText) {
            for (const auto& token : sentence) {
                std::cout << token << ' ';
                if (token == "<eos>") {
                    break;
                }
            }
            std::cout << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "BackpropLab demo failed: " << error.what() << '\n';
        return 1;
    }
}
