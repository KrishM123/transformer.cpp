#ifndef TRANSFORMER_CPP_POSITIONWISEFEEDFORWARD_H
#define TRANSFORMER_CPP_POSITIONWISEFEEDFORWARD_H

#include "utils/Tensor.h"
#include "layers/Linear.h"
#include "layers/Activations.h"
#include <memory>
#include <vector>

class Tensor;
class Linear;
class Activation;
class ReLU;

class PositionwiseFeedForward
{
public:
    // Constructor
    PositionwiseFeedForward(int input_dim, int hidden_dim);

    // Forward pass
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input);

    // Destructor
    ~PositionwiseFeedForward() = default;

private:
    Linear fc1_;
    Linear fc2_;
    ReLU activation_;

    int input_dim_;
    int hidden_dim_;
};

#endif // TRANSFORMER_CPP_POSITIONWISEFEEDFORWARD_H
