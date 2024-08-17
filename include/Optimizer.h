#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#pragma once
#include "../include/Tensor.h"
#include <vector>

/**
 * @class Optimizer
 * @brief Abstract base class for all optimizers used in training neural networks.
 *
 * This class provides a generic interface for optimizers, which are responsible
 * for updating model parameters based on the computed gradients during training.
 *
 * @tparam T Data type used for computations (e.g., float, double).
 */
template <typename T>
class Optimizer {
public:
    /**
     * @class LearningRateSchedule
     * @brief Abstract base class for learning rate schedules.
     *
     * This class defines the interface for learning rate schedules that adjust
     * the learning rate during training based on the epoch number.
     */
    class LearningRateSchedule {
    public:
        /**
         * @brief Get the learning rate for the given epoch.
         *
         * @param epoch Current epoch number.
         * @return T The learning rate for the current epoch.
         */
        virtual T getLearningRate(size_t epoch) = 0;

        virtual ~LearningRateSchedule() = default;

        class StepDecaySchedule;      ///< Step decay schedule for learning rate adjustment.
        class ExponentialDecaySchedule; ///< Exponential decay schedule for learning rate adjustment.
    };

    /**
     * @brief Constructor for the Optimizer class.
     *
     * @param beta1 Coefficient used for computing running averages of gradient.
     * @param beta2 Coefficient used for computing running averages of squared gradient.
     * @param epsilon Small constant for numerical stability.
     * @param learning_rate Initial learning rate.
     * @param lr_schedule Reference to the learning rate schedule.
     */
    explicit Optimizer(T beta1, T beta2, T epsilon, T learning_rate, LearningRateSchedule& lr_schedule)
     : beta1(beta1), beta2(beta2), epsilon(epsilon), learning_rate(learning_rate), lr_schedule(lr_schedule) {}

    /**
     * @brief Initialize parameters for the optimizer.
     *
     * @param param_shape Shapes of the parameters to be optimized.
     */
    virtual void initialize_params(const std::vector<std::vector<int>>& param_shape) = 0;

    /**
     * @brief Update the model parameters based on gradients.
     *
     * @param params References to the model parameters.
     * @param grads References to the gradients of the model parameters.
     * @param epoch Current epoch number.
     */
    virtual void update(const std::vector<std::reference_wrapper<Tensor<T>>>& params, const std::vector<std::reference_wrapper<Tensor<T>>>& grads, const size_t& epoch) = 0;

    virtual ~Optimizer() = default;

    class Adam;    ///< Adam optimizer implementation.
    class RMSprop; ///< RMSprop optimizer implementation.
    class SGD;     ///< Stochastic Gradient Descent (SGD) optimizer implementation.

protected:
    T beta1; ///< Coefficient for computing running averages of gradient.
    T beta2; ///< Coefficient for computing running averages of squared gradient.
    T epsilon; ///< Small constant for numerical stability.

    size_t time_step = 0; ///< Current time step.
    std::vector<Tensor<T>> first_moment_vector; ///< First moment vectors for the parameters.
    std::vector<Tensor<T>> second_moment_vector; ///< Second moment vectors for the parameters.
    Tensor<T> mean_squared_gradients; ///< Mean squared gradients for RMSprop optimizer.

    T learning_rate; ///< Current learning rate.
    LearningRateSchedule& lr_schedule; ///< Reference to the learning rate schedule.
    T decay_rate; ///< Decay rate for RMSprop and other decay-based optimizers.

    /**
     * @brief Update the learning rate based on the current epoch.
     *
     * @param epoch Current epoch number.
     */
    void updateLearningRate(size_t epoch);

    /**
     * @brief Apply weight decay to the parameters.
     *
     * @param params Tensor containing the parameters.
     * @param weight_decay Decay factor.
     */
    static void apply_weight_decay(Tensor<T>& params, T weight_decay);

    /**
     * @brief Clip the gradients to a specified range.
     *
     * @param grads Tensor containing the gradients.
     * @param clip_value Maximum allowed value for gradients.
     */
    void clip_gradients(Tensor<T>& grads, T clip_value);

    /**
     * @brief Save the state of the optimizer to a file.
     *
     * @param filename Name of the file to save the state to.
     */
    void save_state(const std::string& filename) const;

    /**
     * @brief Load the state of the optimizer from a file.
     *
     * @param filename Name of the file to load the state from.
     */
    void load_state(const std::string& filename);

    /**
     * @brief Deserialize the optimizer state from an input stream.
     *
     * @param is Input stream to deserialize from.
     */
    void deserialize(std::istream& is);
};

#include "../src/Optimizer.tpp"

#endif // OPTIMIZER_H
