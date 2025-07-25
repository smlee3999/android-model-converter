# original_dataset_solution.py
# Complete neural network model conversion using original dataset

import os
import warnings
import numpy as np
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import original assignment files
try:
    from dnn_app_utils_v3 import *
    print("Original assignment files imported successfully")
except ImportError as e:
    print(f"Failed to import original files: {e}")
    print("Required files:")
    print("   - dnn_app_utils_v3.py")
    exit(1)

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

print("Neural Network Model Conversion using Original Dataset")
print("="*60)

# ==================== Original Assignment Functions ====================
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    grads -- dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    # compute activation
    A = sigmoid(np.dot(w.T, X)+b)
    # compute cost by using np.dot to perform multiplication.
    cost = -np.mean(Y*np.log(A[0]) + (1-Y)*np.log(1-A[0]))

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X, (A[0]-Y).T)/X.shape[1]
    db = np.mean(A[0]-Y)

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads,cost = propagate(w,b,X,Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def lr_predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X)+b)

    for i in range(A[0].shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0][0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i]=0

    return Y_prediction

def lr_model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zeros
    w = np.zeros((X_train.shape[0], 1))
    b = 0.0

    # Gradient descent
    params, grads, costs= optimize( w, b, X_train, Y_train,num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "params"
    w = params['w']
    b = params['b']

    # # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_train = lr_predict(w, b, X_train)
    Y_prediction_test = lr_predict(w, b, X_test)

    # # Print train/test Errors
    # if print_cost:
    #     print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    #     print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")

        # Compute cost
        cost = compute_cost(A2, Y)

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 iterations and for the last iteration
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)

    return parameters, costs

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations and for the last iteration
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)

    return parameters, costs

# ==================== TensorFlow Conversion Class ====================
class OriginalModelConverter:
    """Convert original assignment NumPy model to TensorFlow Lite"""
    
    def __init__(self, numpy_model, numpy_parameters, layers_dims = []):
        self.numpy_model = numpy_model
        self.layers_dims = layers_dims
        self.numpy_parameters = numpy_parameters
        self.tf_model = None
        print(f"{self.numpy_model}: Original model converter initialized")

    def create_tensorflow_model(self):
        """Create TensorFlow model with same structure as original"""
        print(f"{self.numpy_model}: Creating TensorFlow model (original structure)...")
        
        model = keras.Sequential([
            layers.Input(shape=(64 * 64 * 3,), name='input_layer'),
        ])

        L = len(self.layers_dims)            # number of layers in the network
        for l in range(1, L-1):
            model.add(layers.Dense(self.layers_dims[l], activation='relu', name='hidden_layer' + str(l)))
        model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))
    
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.tf_model = model
        print(f"{self.numpy_model}: TensorFlow model creation completed")
        return model

    def transfer_weights(self):
        """Transfer original NumPy weights to TensorFlow model"""
        print(f"{self.numpy_model}: Transferring original weights...")

        if self.tf_model is None:
            raise ValueError("Please create TensorFlow model first!")

        tf_parameters = {}
        L = len(self.layers_dims)            # number of layers in the network

        if (L == 0):
            tf_parameters["W1"] = self.numpy_parameters["W1"]
            tf_parameters["b1"] = self.numpy_parameters["b1"].flatten()

            self.tf_model.get_layer('output_layer').set_weights([tf_parameters["W1"], tf_parameters["b1"]])
        else:
            for l in range(1, L):
                tf_parameters["W" + str(l)] = self.numpy_parameters["W" + str(l)].T
                tf_parameters["b" + str(l)] = self.numpy_parameters["b" + str(l)].flatten()

            for l in range(1, L - 1):
                self.tf_model.get_layer('hidden_layer' + str(l)).set_weights([tf_parameters["W" + str(l)], tf_parameters["b" + str(l)]])
            self.tf_model.get_layer('output_layer').set_weights([tf_parameters["W" + str(L-1)], tf_parameters["b" + str(L-1)]])

        print(f"{self.numpy_model}: Original weights transfer completed")

    def convert_to_tflite(self, filename='original_planar_classifier.tflite'):
        """Convert to TensorFlow Lite"""
        print(f"{self.numpy_model}: Converting to TensorFlow Lite...")
        
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
    print(f"Testing model with original data: \033[33m{tflite_filename}\033[0m")
    
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
        print("\n\033[32m  #1. Load and Process the Dataset\033[0m")
        train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

        # Explore your dataset
        m_train = train_x_orig.shape[0]
        num_px = train_x_orig.shape[1]
        m_test = test_x_orig.shape[0]

        print ("Number of training examples: " + str(m_train))
        print ("Number of testing examples:  " + str(m_test))
        print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print ("   train_x_orig shape: " + str(train_x_orig.shape))
        print ("   train_y shape:      " + str(train_y.shape))
        print ("   test_x_orig shape:  " + str(test_x_orig.shape))
        print ("   test_y shape:       " + str(test_y.shape))

        # Reshape the training and test examples
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        # Standardize data to have feature values between 0 and 1.
        train_x = train_x_flatten/255.
        test_x = test_x_flatten/255.

        print ("   train_x's shape:    " + str(train_x.shape))
        print ("   test_x's shape:     " + str(test_x.shape))

        # 2. Train original neural network model
        print("\n\033[32m  #2. Train the model\033[0m")

        print("\n\033[34m[W2] Logistic Regression\033[0m")
        lr_parameters = lr_model(train_x, train_y, test_x, test_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

        print("\n\033[34m[W4] 2-Layer Neural Network\033[0m")
        tl_layers_dims = (12288, 7, 1)
        tl_parameters, tl_costs = two_layer_model(train_x, train_y, layers_dims = tl_layers_dims, num_iterations = 2500, print_cost=True)

        print("\n\033[34m[W4] L-Layer Neural Network\033[0m")
        ll_layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
        ll_parameters, ll_costs = L_layer_model(train_x, train_y, ll_layers_dims, num_iterations = 2500, print_cost = True)

        # 3. Evaluate original model performance
        print("\n\033[32m  #3. Evaluating original NumPy model performance\033[0m")
        # Predict test/train set examples (≈ 2 lines of code)
        lr_predictions_train = lr_predict(lr_parameters["w"], lr_parameters["b"], train_x)
        lr_predictions_test = lr_predict(lr_parameters["w"], lr_parameters["b"], test_x)

        print("\033[34m[W2] Logistic Regression\033[0m")
        print(" train accuracy: {} %".format(100 - np.mean(np.abs(lr_predictions_train - train_y)) * 100))
        print("  test accuracy: {} %".format(100 - np.mean(np.abs(lr_predictions_test - test_y)) * 100))

        print("\033[34m[W4] 2-Layer Neural Network\033[0m")
        print(" train ", end='')
        tl_predictions_train = predict(train_x, train_y, tl_parameters)
        print("  test ", end='')
        tl_predictions_test = predict(test_x, test_y, tl_parameters)

        print("\033[34m[W4] L-Layer Neural Network\033[0m")
        print(" train ", end='')
        tl_predictions_train = predict(train_x, train_y, ll_parameters)
        print("  test ", end='')
        tl_predictions_test = predict(test_x, test_y, ll_parameters)

        # 4. TensorFlow conversion
        print("\n\033[32m  #4. Starting original model TensorFlow conversion\033[0m")
        lr_converter = OriginalModelConverter("\033[34m[W2]    Logistic Regression\033[0m", {"W1": lr_parameters["w"], "b1": lr_parameters["b"]})
        tf_model = lr_converter.create_tensorflow_model()
        lr_converter.transfer_weights()

        tl_converter = OriginalModelConverter("\033[34m[W4] 2-Layer Neural Network\033[0m", tl_parameters, tl_layers_dims)
        tf_model = tl_converter.create_tensorflow_model()
        tl_converter.transfer_weights()

        ll_converter = OriginalModelConverter("\033[34m[W4] L-Layer Neural Network\033[0m", ll_parameters, ll_layers_dims)
        tf_model = ll_converter.create_tensorflow_model()
        ll_converter.transfer_weights()

        # 5. TensorFlow Lite conversion
        lr_tflite_model, lr_tflite_filename = lr_converter.convert_to_tflite("logistic_regression.tflite")
        tl_tflite_model, tl_tflite_filename = tl_converter.convert_to_tflite("nn_two_layer_model.tflite")
        ll_tflite_model, ll_tflite_filename = ll_converter.convert_to_tflite("nn_l_layer_model.tflite")
        
        if lr_tflite_model is None or tl_tflite_model is None or ll_tflite_model is None:
            print("TensorFlow Lite conversion failed")
            return False
        
        # 6. Test TensorFlow Lite model
        print("\n\033[32m  #6. Testing TensorFlow Lite model with original data\033[0m")
        tflite_predictions, tflite_accuracy = test_original_tflite_model(lr_tflite_filename, train_x, train_y)
        tflite_predictions, tflite_accuracy = test_original_tflite_model(lr_tflite_filename, test_x, test_y)
        tflite_predictions, tflite_accuracy = test_original_tflite_model(tl_tflite_filename, train_x, train_y)
        tflite_predictions, tflite_accuracy = test_original_tflite_model(tl_tflite_filename, test_x, test_y)
        tflite_predictions, tflite_accuracy = test_original_tflite_model(ll_tflite_filename, train_x, train_y)
        tflite_predictions, tflite_accuracy = test_original_tflite_model(ll_tflite_filename, test_x, test_y)
        
        # 7. Visualize in original style
        # print("\nVisualizing results in original style...")
        # visualize_original_results(X, Y, parameters, tflite_predictions, tflite_filename)
        
        # 8. Create Android files
        # print("\nCreating Android files...")
        # create_android_files_original(tflite_filename, X, Y)
        
        # 9. Final summary
        print("\n" + "="*60)
        print("All processes completed with original dataset!")
        print("="*60)
        # print(f"Original NumPy model accuracy: {accuracy:.2f}%")
        # if tflite_predictions is not None:
        #     print(f"TensorFlow Lite accuracy: {tflite_accuracy*100:.2f}%")
        # print(f"TensorFlow version: {tf.__version__}")
        # print(f"Number of data points: {X.shape[1]}")
        
        # print(f"\nGenerated files:")
        # files = [
        #     tflite_filename,
        #     'original_dataset.png',
        #     'original_model_comparison.png',
        #     'original_android_config.json',
        #     'original_android_guide.md'
        # ]
        
        # for file in files:
        #     if os.path.exists(file):
        #         size = os.path.getsize(file)
        #         print(f"  {file} ({size} bytes)")
        
        # print(f"\nAndroid development ready with original data!")
        # print(f"   Main model file: {tflite_filename}")
        
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
    # print("   (Requires: testCases_v2.py, public_tests.py, planar_utils.py)")
    print()
    
    success = main_with_original_dataset()
    if success:
        print("\nOriginal data-based conversion completed successfully!")
    else:
        print("\nProblems occurred during execution.")
