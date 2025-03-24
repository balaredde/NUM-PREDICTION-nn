# Neural Network for MNIST Classification

## Description
This project implements a neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model is trained on a dataset of 60,000 images and tested on 10,000 images.

## Installation
To run this project, install the required dependencies:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Usage
Run the Jupyter Notebook to train and test the model:

```bash
jupyter notebook neural_network.ipynb
```

## Dataset
The project uses the **MNIST dataset**, which consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is loaded using Keras:

```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

## Technologies Used
- **TensorFlow & Keras** - For building and training the neural network
- **NumPy** - For numerical operations
- **Pandas** - For data manipulation
- **Matplotlib** - For visualizing data
- **Scikit-learn** - For additional preprocessing

## Author
[Your Name]

## License
This project is licensed under the MIT License.
