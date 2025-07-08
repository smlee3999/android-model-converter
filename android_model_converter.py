# original_dataset_solution.py
# Complete neural network model conversion using original assignment planar_utils dataset

import os
import warnings
import numpy as np
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

# Import original assignment files
try:
    from testCases_v2 import *
    from public_tests import *
    from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
    print("Original assignment files imported successfully")
except ImportError as e:
    print(f"Failed to import original files: {e}")
    print("Required files:")
    print("   - testCases_v2.py")
    print("   - public_tests.py") 
    print("   - planar_utils.py")
    exit(1)

import sklearn
import sklearn.datasets
import sklearn.linear_model

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

print("Neural Network Model Conversion using Original Dataset")
print("="*60)

# ==================== Original Assignment Functions ====================
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]   # Number of input features
    n_h = 4            # Hidden layer neurons fixed to 4
    n_y = Y.shape[0]   # Number of output neurons
    
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)
    
    W1 = np.random.randn(n_h, n_x) * 0.01  # Small random values
    b1 = np.zeros((n_h, 1))                # Initialize bias to zero
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)             # Non-linear activation function
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)             # Sigmoid output (probability)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    m = Y.shape[1] # number of examples
    
    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -np.sum(logprobs) / m
    
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
                                    # E.g., turns [[17]] into 17
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients
    
    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = copy.deepcopy(parameters["W1"])
    b1 = copy.deepcopy(parameters["b1"])
    W2 = copy.deepcopy(parameters["W2"])
    b2 = copy.deepcopy(parameters["b2"])
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Loop (gradient descent)
    costs = []
    for i in range(0, num_iterations):
        
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y". Outputs: "cost".
        cost = compute_cost(A2, Y)
        
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
        
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate=1.2)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
    
    return parameters, costs

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

# ==================== TensorFlow Conversion Class ====================
class OriginalModelConverter:
    """Convert original assignment NumPy model to TensorFlow Lite"""
    
    def __init__(self, numpy_parameters):
        self.numpy_parameters = numpy_parameters
        self.tf_model = None
        print("Original model converter initialized")
    
    def create_tensorflow_model(self):
        """Create TensorFlow model with same structure as original"""
        print("Creating TensorFlow model (original structure)...")
        
        model = keras.Sequential([
            layers.Input(shape=(2,), name='input_layer'),
            layers.Dense(4, activation='tanh', name='hidden_layer'),
            layers.Dense(1, activation='sigmoid', name='output_layer')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.tf_model = model
        print("TensorFlow model creation completed")
        return model
    
    def transfer_weights(self):
        """Transfer original NumPy weights to TensorFlow model"""
        print("Transferring original weights...")
        
        if self.tf_model is None:
            raise ValueError("Please create TensorFlow model first!")
        
        # Extract original NumPy parameters
        W1 = self.numpy_parameters['W1']  # (4, 2)
        b1 = self.numpy_parameters['b1']  # (4, 1)
        W2 = self.numpy_parameters['W2']  # (1, 4)
        b2 = self.numpy_parameters['b2']  # (1, 1)
        
        # Convert to TensorFlow format (transpose required)
        tf_W1 = W1.T  # (2, 4)
        tf_b1 = b1.flatten()  # (4,)
        tf_W2 = W2.T  # (4, 1)
        tf_b2 = b2.flatten()  # (1,)
        
        # Set weights
        self.tf_model.get_layer('hidden_layer').set_weights([tf_W1, tf_b1])
        self.tf_model.get_layer('output_layer').set_weights([tf_W2, tf_b2])
        
        print("Original weights transfer completed")
    
    def convert_to_tflite(self, filename='original_planar_classifier.tflite'):
        """Convert to TensorFlow Lite"""
        print("Converting to TensorFlow Lite...")
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Perform conversion
        try:
            tflite_model = converter.convert()
            
            # Save file
            with open(filename, 'wb') as f:
                f.write(tflite_model)
            
            file_size = len(tflite_model)
            print(f"TensorFlow Lite conversion completed!")
            print(f"   File: {filename}")
            print(f"   Size: {file_size} bytes")
            
            return tflite_model, filename
            
        except Exception as e:
            print(f"Conversion failed: {e}")
            return None, None

# ==================== Model Validation ====================
def test_original_tflite_model(tflite_filename, X_test, Y_test):
    """Test TensorFlow Lite model with original data"""
    print(f"Testing model with original data: {tflite_filename}")
    
    try:
        # Create interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_filename)
        interpreter.allocate_tensors()
        
        # Input/Output information
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        
        # Prepare test data (keep original data format)
        test_input = X_test.T.astype(np.float32)  # (m, 2)
        predictions = []
        
        # Predict for each sample
        for i in range(test_input.shape[0]):
            # Set input
            interpreter.set_tensor(input_details[0]['index'], test_input[i:i+1])
            
            # Run inference
            interpreter.invoke()
            
            # Get result
            output = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output[0][0])
        
        # Calculate accuracy
        predictions = np.array(predictions)
        binary_predictions = (predictions > 0.5).astype(int)
        accuracy = np.mean(binary_predictions == Y_test.flatten())
        
        print(f"Original data test completed! Accuracy: {accuracy*100:.2f}%")
        
        return predictions, accuracy
        
    except Exception as e:
        print(f"Test failed: {e}")
        return None, 0

# ==================== Visualization (using original functions) ====================
def visualize_original_results(X, Y, parameters, tflite_predictions=None, tflite_filename=None):
    """Visualize results using original assignment's plot_decision_boundary function"""
    print("Visualizing results in original style...")
    
    try:
        plt.figure(figsize=(16, 6))
        
        # 1. Original data
        plt.subplot(1, 3, 1)
        plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
        plt.title("Original Dataset")
        
        # 2. NumPy model decision boundary (using original function)
        plt.subplot(1, 3, 2)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        plt.title("NumPy Neural Network")
        
        # Calculate NumPy accuracy
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
        print(f"   NumPy model accuracy: {accuracy:.1f}%")
        
        # 3. TensorFlow Lite prediction results
        plt.subplot(1, 3, 3)
        if tflite_predictions is not None:
            tflite_binary = (tflite_predictions > 0.5).astype(int)
            plt.scatter(X[0, :], X[1, :], c=tflite_binary, s=40, cmap=plt.cm.Spectral)
            tflite_acc = np.mean(tflite_binary == Y.flatten()) * 100
            plt.title(f'TensorFlow Lite\nAccuracy: {tflite_acc:.1f}%')
        else:
            plt.title('TensorFlow Lite\n(Not Available)')
        plt.xlabel('X1')
        plt.ylabel('X2')
        
        plt.tight_layout()
        plt.savefig('original_model_comparison.png', dpi=150, bbox_inches='tight')
        
        print("Original style visualization completed: original_model_comparison.png")
        
        try:
            plt.show()
        except:
            print("   (No GUI - saved as image file only)")
            
    except Exception as e:
        print(f"Visualization failed: {e}")

# ==================== Main Execution ====================
def main_with_original_dataset():
    """Complete process using original dataset"""
    print("Starting complete process with original assignment data!")
    
    try:
        # 1. Load original data
        print("\nLoading original planar dataset...")
        X, Y = load_planar_dataset()
        print(f"Original data loaded successfully: X={X.shape}, Y={Y.shape}")
        
        # Visualize data (original style)
        print("Visualizing original data...")
        plt.figure(figsize=(6, 4))
        plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
        plt.title("Original Planar Dataset")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.savefig('original_dataset.png', dpi=150, bbox_inches='tight')
        print("Original data visualization saved: original_dataset.png")
        
        # 2. Train original neural network model
        print("\nStarting original neural network model training...")
        parameters, costs = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
        
        # 3. Evaluate original model performance
        print("\nEvaluating original NumPy model performance...")
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
        print(f"Original NumPy model accuracy: {accuracy:.2f}%")
        
        # 4. TensorFlow conversion
        print("\nStarting original model TensorFlow conversion...")
        converter = OriginalModelConverter(parameters)
        tf_model = converter.create_tensorflow_model()
        converter.transfer_weights()
        
        # 5. TensorFlow Lite conversion
        tflite_model, tflite_filename = converter.convert_to_tflite()
        
        if tflite_model is None:
            print("TensorFlow Lite conversion failed")
            return False
        
        # 6. Test TensorFlow Lite model
        print("\nTesting TensorFlow Lite model with original data...")
        tflite_predictions, tflite_accuracy = test_original_tflite_model(tflite_filename, X, Y)
        
        # 7. Visualize in original style
        print("\nVisualizing results in original style...")
        visualize_original_results(X, Y, parameters, tflite_predictions, tflite_filename)
        
        # 8. Create Android files
        print("\nCreating Android files...")
        create_android_files_original(tflite_filename, X, Y)
        
        # 9. Final summary
        print("\n" + "="*60)
        print("All processes completed with original dataset!")
        print("="*60)
        print(f"Original NumPy model accuracy: {accuracy:.2f}%")
        if tflite_predictions is not None:
            print(f"TensorFlow Lite accuracy: {tflite_accuracy*100:.2f}%")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Number of data points: {X.shape[1]}")
        
        print(f"\nGenerated files:")
        files = [
            tflite_filename,
            'original_dataset.png',
            'original_model_comparison.png',
            'original_android_config.json',
            'original_android_guide.md'
        ]
        
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"  {file} ({size} bytes)")
        
        print(f"\nAndroid development ready with original data!")
        print(f"   Main model file: {tflite_filename}")
        
        return True
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_android_files_original(tflite_filename, X, Y):
    """Create Android files based on original dataset"""
    print("Creating Android files based on original data...")
    
    # Calculate data ranges
    x_min, x_max = float(X[0, :].min()), float(X[0, :].max())
    y_min, y_max = float(X[1, :].min()), float(X[1, :].max())
    
    # Android configuration file
    android_config = {
        "model_name": "OriginalPlanarClassifier",
        "version": "1.0",
        "description": "Original course assignment planar data classification",
        "dataset_info": {
            "total_points": int(X.shape[1]),
            "features": int(X.shape[0]),
            "x_range": [x_min, x_max],
            "y_range": [y_min, y_max],
            "class_distribution": {
                "red_points": int(np.sum(Y == 0)),
                "blue_points": int(np.sum(Y == 1))
            }
        },
        "model_info": {
            "input": {
                "type": "FLOAT32",
                "shape": [1, 2],
                "description": "X, Y coordinates"
            },
            "output": {
                "type": "FLOAT32", 
                "shape": [1, 1],
                "range": [0.0, 1.0],
                "threshold": 0.5,
                "description": "Probability of blue class"
            }
        },
        "classes": {
            "0": {"name": "red", "color": "#FF0000"},
            "1": {"name": "blue", "color": "#0000FF"}
        }
    }
    
    with open('original_android_config.json', 'w') as f:
        json.dump(android_config, f, indent=2)
    
    # Usage guide
    usage_guide = f"""# Original Planar Dataset Android Integration

## Dataset Information
- **Total Points**: {X.shape[1]}
- **X Range**: [{x_min:.3f}, {x_max:.3f}]  
- **Y Range**: [{y_min:.3f}, {y_max:.3f}]
- **Red Points**: {np.sum(Y == 0)}
- **Blue Points**: {np.sum(Y == 1)}

## Model File
- **Filename**: {tflite_filename}
- **Size**: {os.path.getsize(tflite_filename)} bytes

## Android Implementation

### 1. File Setup
```
app/src/main/assets/{tflite_filename}
```

### 2. Coordinate Normalization
The original dataset has specific ranges. For touch coordinates:

```kotlin
fun normalizeToOriginalRange(touchX: Float, touchY: Float, 
                           screenWidth: Int, screenHeight: Int): Pair<Float, Float> {{
    // Map touch coordinates to original dataset range
    val normalizedX = (touchX / screenWidth) * ({x_max:.3f} - ({x_min:.3f})) + {x_min:.3f}f
    val normalizedY = (touchY / screenHeight) * ({y_max:.3f} - ({y_min:.3f})) + {y_min:.3f}f
    
    return Pair(normalizedX, normalizedY)
}}
```

### 3. Classification
```kotlin
class OriginalPlanarClassifier(context: Context) {{
    private var interpreter: Interpreter? = null
    
    fun classify(x: Float, y: Float): ClassificationResult {{
        val input = floatArrayOf(x, y)
        val output = arrayOf(floatArrayOf(0f))
        
        interpreter?.run(input, output)
        
        val probability = output[0][0]
        val isBlue = probability > 0.5f
        val confidence = if (isBlue) probability else (1f - probability)
        
        return ClassificationResult(
            className = if (isBlue) "blue" else "red",
            probability = probability,
            confidence = confidence
        )
    }}
}}
```

### 4. Data Ranges for Reference
- Original X range: [{x_min:.3f}, {x_max:.3f}]
- Original Y range: [{y_min:.3f}, {y_max:.3f}]

This matches the exact dataset used in the course assignment.
"""
    
    with open('original_android_guide.md', 'w') as f:
        f.write(usage_guide)
    
    print("Android files creation completed based on original data")
    print("   - original_android_config.json")
    print("   - original_android_guide.md")

if __name__ == "__main__":
    print("Starting model conversion using original assignment dataset!")
    print("   (Requires: testCases_v2.py, public_tests.py, planar_utils.py)")
    print()
    
    success = main_with_original_dataset()
    if success:
        print("\nOriginal data-based conversion completed successfully!")
    else:
        print("\nProblems occurred during execution.")