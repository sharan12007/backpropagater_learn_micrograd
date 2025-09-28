# Micrograd Andrej Learn

This project is a minimal neural network and autograd engine inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). It demonstrates how to build a simple multi-layer perceptron (MLP) from scratch using pure Python and NumPy, with automatic differentiation for backpropagation.

## Features

- Custom `Value` class for scalar automatic differentiation
- Fully connected neural network (MLP) with configurable layers
- Manual training loop with gradient descent
- Visualization of computation graph using Graphviz
- Example usage with synthetic data

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Graphviz
- IPython (for Jupyter Notebook display)

Install dependencies:
```bash
pip install numpy matplotlib graphviz ipython
```

## Usage

1. **Define the network:**
    ```python
    n = MLP(3, [4, 4, 1])  # 3 input features, two hidden layers (4 neurons each), 1 output
    ```

2. **Prepare data:**
    ```python
    x = np.array([[2.0, -1.0, 3.0], [1.0, 3.0, -1.0], [2.0, 0.0, 1.0], [5.4, 1.2, 2.7]])
    y = np.array([1.0, -1.0, 1.0, -1.0])
    ```

3. **Training loop:**
    ```python
    for i in range(1000):
        y_pred = [n([Value(v) for v in xi]) for xi in x]
        loss = sum((y_p - Value(y_t)) ** 2 for y_p, y_t in zip(y_pred, y))
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()
        for p in n.parameters():
            p.data += -0.01 * p.grad
    ```

4. **Plot predictions:**
    ```python
    import matplotlib.pyplot as plt
    y_pred_data = [v.data for v in y_pred]
    plt.plot(y, y_pred_data, 'o')
    plt.xlabel('True y')
    plt.ylabel('Predicted y')
    plt.title('True vs Predicted')
    plt.show()
    ```

5. **Visualize computation graph:**
    ```python
    from IPython.display import display
    display(draw_dot(loss))
    ```

## Files

- `micrograd_andrej_learn.ipynb`: Main Jupyter notebook with all code and experiments
- `README.md`: Project documentation

## References

- [micrograd by Andrej Karpathy](https://github.com/karpathy/micrograd)
- [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0)

---

**Note:** This project is for educational purposes and does not use PyTorch or TensorFlow. All autograd and neural network logic is implemented from scratch.