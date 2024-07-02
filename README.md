# Linear Regression With Gradient Descent

This project implements a simple linear regression model using gradient descent optimization from scratch. The code reads data from a CSV file, splits it into training and testing sets, performs gradient descent to optimize the model parameters, and visualizes the results. The project also includes implementations of gradient descent with **L1 and L2 regularization** to prevent overfitting and improve model generalization.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Gradient Descent for Linear Regression](#gradient-descent-for-linear-regression)
- [L1 and L2 Regularization](#l1-and-l2-regularization)
- [Example](#example)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/gradient-descent-linear-regression.git
    cd gradient-descent-linear-regression
    

2. Install the required Python packages:
        pip install -r requirements.txt
    

## Usage

1. Prepare your CSV file with the data. Ensure that the data is clean and in the proper format.

2. Modify the data_url variable in the script to point to your CSV file.

3. Run the script:
        python main.py
    

## Gradient Descent for Linear Regression

### Loss(X, y, W, b)
Calculates the mean squared error loss.

- Parameters:
  - X: Input features matrix.
  - y: Target values.
  - W: Weights.
  - b: Bias.

### dldw(X, y, W, b)
Calculates the gradient of the loss function with respect to the weights.

- Parameters:
  - X: Input features matrix.
  - y: Target values.
  - W: Weights.
  - b: Bias.

### dldb(X, y, W, b)
Calculates the gradient of the loss function with respect to the bias.

- Parameters:
  - X: Input features matrix.
  - y: Target values.
  - W: Weights.
  - b: Bias.

### gradient_descent(X, y, W, b, max_iterations=10000)
Performs gradient descent optimization to minimize the loss function.

- Parameters:
  - X: Input features matrix.
  - y: Target values.
  - W: Initial weights.
  - b: Initial bias.
  - max_iterations: Maximum number of iterations.

### get_beta(X, y, W, b)
Calculates the step size (learning rate) using the Fibonacci method.

- Parameters:
  - X: Input features matrix.
  - y: Target values.
  - W: Weights.
  - b: Bias.

### plot_scatter(X, y, W, b)
Plots the scatter plot of the input features and target values along with the regression line.

- Parameters:
  - X: Input features matrix.
  - y: Target values.
  - W: Weights.
  - b: Bias.

## L1 and L2 Regularization

### L1 Regularization

L1 regularization, also known as Lasso regularization, adds a penalty term to the loss function that is proportional to the sum of the absolute values of the weights (coefficients). This encourages sparsity in the model by pushing some weights to exactly zero, effectively performing feature selection.

### L2 Regularization

L2 regularization, also known as Ridge regularization, adds a penalty term to the loss function that is proportional to the sum of the squared values of the weights (coefficients). This penalizes large weights and encourages smaller weights, which helps prevent overfitting.

### Choosing Between L1 and L2 Regularization

- L1 regularization is useful when the goal is to achieve sparsity in the model, i.e., to identify and select the most important features while discarding less relevant ones.
  
- L2 regularization is effective in preventing overfitting by penalizing large weights, leading to smoother models and better generalization.

In practice, a combination of L1 and L2 regularization (Elastic Net regularization) can be used to leverage the benefits of both approaches.

## Example

The script demonstrates how to use the gradient descent algorithm to fit a linear regression model to the data. It includes the following steps:

1. Load and preprocess the data.
2. Split the data into training and testing sets.
3. Initialize the weights and bias.
4. Perform gradient descent optimization with L1 and L2 regularization.
5. Calculate the loss on the test set.
6. Predict on the test set and visualize the results.

## Output
The script prints the loss on the test set and the execution time. It also generates a scatter plot comparing the actual and predicted values.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
