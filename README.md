# LENS - Light & Easy Neuron Network System

A lightweight neural network framework implemented in Python 3 and NumPy, designed for simplicity and educational purposes.

## Features

- **Pure NumPy Implementation**: No heavy dependencies, just NumPy for numerical computations
- **Modular Architecture**: Clean separation of layers, optimizers, and loss functions
- **Automatic Differentiation**: Built-in backpropagation with gradient computation
- **Configurable Training**: YAML-based configuration for hyperparameters
- **Multiple Activation Functions**: ReLU and Sigmoid activation functions
- **Loss Functions**: Mean Squared Error (MSE) and Cross-Entropy loss
- **SGD Optimizer**: Stochastic Gradient Descent with configurable learning rate

## Installation

### Prerequisites
- Python 3.x
- NumPy
- PyYAML

### Setup
```bash
git clone <repository-url>
cd light_easy_neuronNetwork_system
pip install numpy pyyaml
```

## Quick Start

```python
from lib.lenslib import Lens
import numpy as np

# Prepare your data
X = np.array([[-1,-1,-1,1],
              [-1,-1,1,1],
              [1,1,0,1],
              [1,1,1,1]])
Y = np.array([[-1,-1,1,1]]).T

# Create and train the model
lens = Lens(X.shape)
losses = lens.learn(X, Y)

# Make predictions
prediction = lens.forward([0,0,1,-1])
print(prediction)
```

## Architecture

### Core Components

#### 1. Parameter Class
Manages tensors and their gradients for automatic differentiation.

#### 2. Layer Classes
- **Layer**: Base class for all neural network layers
- **Linear**: Fully connected layer with weights and biases
- **Sequential**: Container for stacking multiple layers
- **ReLU**: Rectified Linear Unit activation function
- **Sigmoid**: Sigmoid activation function

#### 3. Loss Functions
- **mse_loss**: Mean Squared Error for regression tasks
- **ce_loss**: Cross-Entropy loss for classification tasks

#### 4. Optimizer
- **SGDOptimizer**: Stochastic Gradient Descent optimizer

#### 5. Learner Class
High-level training interface that combines model, loss function, and optimizer.

#### 6. Lens Class
Main interface that provides a simple API for creating and training neural networks.

## Configuration

The system uses a YAML configuration file (`lib/lensconf.yaml`) for hyperparameters:

```yaml
epochs: 10
batch_size: 1
learning_rate: 0.05
value_range: 2
value_diff: -1
```

### Configuration Parameters
- `epochs`: Number of training epochs
- `batch_size`: Size of training batches
- `learning_rate`: Learning rate for SGD optimizer
- `value_range`: Range for weight initialization
- `value_diff`: Offset for weight initialization

## API Reference

### Lens Class

#### Constructor
```python
Lens(dataset_shape)
```
- `dataset_shape`: Tuple of (num_samples, num_features)

#### Methods

##### learn(X, Y)
Train the neural network on the provided data.
- `X`: Input features (numpy array)
- `Y`: Target values (numpy array)
- Returns: List of losses for each epoch

##### forward(X)
Make predictions on new data.
- `X`: Input features
- Returns: Predicted values

### Building Custom Models

```python
from lib.lenslib import Sequential, Linear, ReLU, Sigmoid, Learner, mse_loss, SGDOptimizer

# Create a custom model
model = Sequential(
    Linear(4, 8, value_range=2, value_diff=-1),
    ReLU(),
    Linear(8, 1, value_range=2, value_diff=-1),
    Sigmoid()
)

# Create learner
learner = Learner(model, mse_loss, SGDOptimizer(lr=0.01))

# Train
losses = learner.fit(X, Y, epochs=100, bs=32)
```

## Examples

### Basic Linear Regression
```python
from lib.lenslib import Lens
import numpy as np

# Generate sample data
X = np.random.randn(100, 3)
Y = X @ np.array([1.5, -2.0, 0.5]) + np.random.randn(100) * 0.1

# Train model
lens = Lens(X.shape)
losses = lens.learn(X, Y.reshape(-1, 1))

# Make predictions
test_input = np.array([[1.0, 0.5, -0.3]])
prediction = lens.forward(test_input)
```

### Multi-layer Neural Network
```python
from lib.lenslib import Sequential, Linear, ReLU, Learner, mse_loss, SGDOptimizer

# Create multi-layer model
model = Sequential(
    Linear(10, 64, value_range=2, value_diff=-1),
    ReLU(),
    Linear(64, 32, value_range=2, value_diff=-1),
    ReLU(),
    Linear(32, 1, value_range=2, value_diff=-1)
)

learner = Learner(model, mse_loss, SGDOptimizer(lr=0.001))
```

## File Structure

```
light_easy_neuronNetwork_system/
├── lib/
│   ├── __init__.py          # Package initialization
│   ├── lensconf.yaml        # Configuration file
│   ├── lenslib.py          # Main neural network library
│   └── yamllib.py          # YAML configuration reader
├── test_case.py            # Example usage and test case
└── README.md               # This file
```

## Testing

Run the included test case:
```bash
python test_case.py
```

This will train a simple neural network on a small dataset and display the training losses and a sample prediction.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Built with NumPy for efficient numerical computations
- Inspired by modern deep learning frameworks but simplified for educational purposes
- Designed to demonstrate core neural network concepts without heavy abstractions
