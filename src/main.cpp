#include "../examples/MultiHeadAttentionModel.h"
#include "../include/LossFunction.h"
#include "../include/Optimizer.h"

int main() {
    // Test Multi_head_attention
    // Constants for the test
    constexpr int MAX_SEQUENCE_LENGTH = 64;
    constexpr int HIDDEN_DIM = 64;
    constexpr int NUM_HEADS = 8;
    constexpr int HEAD_DIM = HIDDEN_DIM / NUM_HEADS;
    constexpr int BATCH_SIZE = 8; // Define the batch size
    constexpr float LEARNING_RATE = 0.01;
    constexpr float DECAY_RATE = 0.6;

    // Define the loss function
    LossFunction<float>::meanSquaredError loss_function;  // Ensure your loss function is appropriate for your task

    // Define the learning rate scheduler
    Optimizer<float>::LearningRateSchedule::StepDecaySchedule learning_rate_scheduler(LEARNING_RATE, DECAY_RATE, 1);

    // Define the Multi-Head Attention model
    MultiHeadAttention<float> model(HIDDEN_DIM, NUM_HEADS, HEAD_DIM, new ActivationFunction<float>::ReLU());

    // Collect model parameters for optimization
    std::vector<std::vector<int>> params_shape;
    for (const auto& ref : model.parameters()) {
        auto param_shape = ref.get().shape();
        params_shape.push_back(param_shape);
    }

    // Define the optimizer
    Optimizer<float>::Adam optimizer(params_shape, LEARNING_RATE, learning_rate_scheduler, 1e-8);

    // Define large dummy data for training
    std::vector<Tensor<float>> input_data(1000);
    std::vector<Tensor<float>> target_data(1000);

    // Initialize input and target data with random values
    for (int i = 0; i < 1000; ++i) {
        input_data[i] = Tensor<float>::zeros({MAX_SEQUENCE_LENGTH, HIDDEN_DIM});
        target_data[i] = Tensor<float>::ones({MAX_SEQUENCE_LENGTH, HIDDEN_DIM});
    }

    // Training loop
    for (int epoch = 0; epoch < 5; ++epoch) {
        float loss = 0;
        for (int i = 0; i < 1000; ++i) {
            // Perform forward pass
            Tensor<float> output_train = model.forward(input_data[i]);

            // Compute loss
            loss += loss_function.forward(output_train, target_data[i]);

            // Perform backward pass
            Tensor<float> grad = loss_function.backward(output_train, target_data[i]);
            model.backward(grad);

            optimizer.update(model.parameters(), model.gradients(), epoch);
        }
        std::cout << "Epoch [" << epoch + 1 << "], Loss: " << loss / 1000 << std::endl;
    }

    // Predict with the trained model
    Tensor<float> input_test = Tensor<float>::uniform({MAX_SEQUENCE_LENGTH, HIDDEN_DIM}, 0, 1);
    Tensor<float> output_test = model.forward(input_test);
    output_test.print();
    return 0;
}
