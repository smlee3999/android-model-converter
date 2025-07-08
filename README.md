# Android Model Converter

Neural network model converter for Android deployment from Coursera Deep Learning assignments.

## How It Works

### 🔄 Main Conversion Flow

The converter follows a systematic process to transform Coursera assignment NumPy models into Android-ready TensorFlow Lite models:

```
1. Load Dataset → 2. Train NumPy Model → 3. Create TF Model → 4. Transfer Weights → 5. Convert to TFLite → 6. Validate → 7. Generate Android Files
```

### 📋 Detailed Process

#### 1. **Original Assignment Implementation**
```python
# Uses original Coursera functions:
- layer_sizes()           # Define network architecture (2→4→1)
- initialize_parameters() # Initialize weights and biases
- forward_propagation()   # Compute predictions (tanh + sigmoid)
- backward_propagation()  # Compute gradients
- nn_model()             # Training loop (10,000 iterations)
```

#### 2. **Data Loading & Training**
```python
# Load original planar dataset
X, Y = load_planar_dataset()  # From planar_utils.py

# Train neural network (exactly like Coursera assignment)
parameters, costs = nn_model(X, Y, n_h=4, num_iterations=10000)
# Result: Trained weights {W1, b1, W2, b2}
```

#### 3. **TensorFlow Model Creation**
```python
# Create equivalent TensorFlow model
model = keras.Sequential([
    layers.Dense(4, activation='tanh'),    # Hidden layer (4 neurons)
    layers.Dense(1, activation='sigmoid')  # Output layer (1 neuron)
])
```

#### 4. **Weight Transfer (Key Step)**
```python
# Convert NumPy weights to TensorFlow format
tf_W1 = numpy_W1.T        # Transpose (4,2) → (2,4)
tf_b1 = numpy_b1.flatten() # Flatten (4,1) → (4,)

# Set weights directly (no retraining!)
model.get_layer('hidden_layer').set_weights([tf_W1, tf_b1])
model.get_layer('output_layer').set_weights([tf_W2, tf_b2])
```

#### 5. **TensorFlow Lite Conversion**
```python
# Convert to mobile-optimized format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save as .tflite file
with open('original_planar_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### 6. **Model Validation**
```python
# Test TFLite model accuracy
interpreter = tf.lite.Interpreter(model_path='model.tflite')
# Run inference on test data
# Compare with original NumPy model results
```

#### 7. **Android Integration Files**
```python
# Generate configuration files
- original_android_config.json  # Model metadata
- original_android_guide.md     # Usage instructions
- Visualization images          # Result comparison charts
```

### 🎯 Key Features

**📚 Educational Purpose**
- Uses original Coursera assignment code structure
- Demonstrates NumPy → TensorFlow conversion
- Shows weight transfer without retraining

**🚀 Performance Optimized**
- No model retraining required (instant conversion)
- Preserves exact same accuracy as original
- Generates lightweight mobile models (~2KB)

**📱 Android Ready**
- Creates `.tflite` file for Android assets
- Provides coordinate normalization code
- Includes complete integration guide

### 🔬 Technical Details

**Neural Network Architecture:**
```
Input Layer:    2 neurons (x, y coordinates)
Hidden Layer:   4 neurons (tanh activation)
Output Layer:   1 neuron  (sigmoid activation)
Total Parameters: 13 (2×4 + 4 + 4×1 + 1)
```

**Coordinate Range:**
- **X Range**: [-4.5, 4.0] (from original dataset)
- **Y Range**: [-4.0, 4.0] (from original dataset)
- **Output**: [0.0, 1.0] (probability, threshold=0.5)

**Model Size:**
- **NumPy**: Memory only
- **TensorFlow Lite**: ~2KB file
- **Accuracy**: >90% (identical to original)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

### Coursera Assignment Files (Required)
You need to obtain these files from your Coursera Deep Learning course:
- `testCases_v2.py`
- `public_tests.py`
- `planar_utils.py`

**⚠️ Note**: These files are not included due to copyright restrictions. Download them from your Coursera course and place them in the same directory as the converter script.

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/hyleemarusys/android-model-converter.git
cd android-model-converter
```

### 2. Setup Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify activation (should show (venv) prefix)
which python
```

### 3. Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

### 4. Setup Coursera Files
Download the following files from your Coursera Deep Learning course and place them in the project directory:
- `testCases_v2.py`
- `public_tests.py`  
- `planar_utils.py`

### 5. Run Converter
```bash
# Make sure venv is activated (should see (venv) prefix)
python android_model_converter.py
```

### 6. Deactivate Environment (when done)
```bash
deactivate
```

## Usage

### Basic Conversion
```bash
python android_model_converter.py
```

### Output Files
- `original_planar_classifier.tflite` - TensorFlow Lite model
- `original_android_config.json` - Android configuration
- `original_android_guide.md` - Integration guide

## Model Details

- **Input**: 2D coordinates (x, y)
- **Output**: Binary classification probability
- **Architecture**: 2 → 4 → 1 neural network
- **Activation**: tanh (hidden), sigmoid (output)

## Android Integration

Place the generated `.tflite` file in your Android assets folder:
```
app/src/main/assets/original_planar_classifier.tflite
```

## Project Structure

```
android-model-converter/
├── android_model_converter.py    # Main converter script
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Performance

- Model size: ~2KB
- Inference speed: <1ms on modern devices
- Accuracy: >90% on test dataset

## 🔧 Troubleshooting

### Common Issues

**Virtual Environment not activated**
```bash
# Check if (venv) appears in your prompt
# If not, activate:
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows
```

**TensorFlow installation issues**
```bash
# Try upgrading pip first
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow>=2.10.0,<2.16.0
```

**Missing Coursera files**
```bash
# Ensure files are in the correct location:
ls testCases_v2.py public_tests.py planar_utils.py

# Files should be in the same directory as android_model_converter.py
```

### Environment Reset
```bash
# If something goes wrong, reset everything:
rm -rf venv/              # Remove virtual environment
python3 -m venv venv      # Create new environment
source venv/bin/activate  # Activate
pip install -r requirements.txt  # Reinstall dependencies
```

## License

MIT License - Free to use, modify, and distribute.