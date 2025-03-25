# 🔍 Object Detection Training Pipeline

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A complete pipeline for training object detection models using TensorFlow, focused on the Open Images dataset format.

## ✨ Features

- 📊 Support for Open Images dataset format
- 🚀 GPU acceleration with memory management
- 🔄 Data augmentation and preprocessing
- 🧠 EfficientNetB3-based detection model
- ⚡ Mixed precision training
- 📈 Visualization tools for debugging
- 📱 Export to deployment-ready formats (TF SavedModel, TFLite, TensorRT)

## 🛠️ Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

## 📥 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/jeronimorepetto/object-detection-pipeline.git
   cd object-detection-pipeline
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn tqdm
   ```

## 📂 Data Preparation

The pipeline expects data in the Open Images format with the following directory structure:
```bash
/path/to/your/data/
├── target_dir/
│ ├── train/
│ ├── validation/
│ └── test/
├── boxes/
│ ├── oidv6-train-annotations-bbox.csv
│ ├── validation-annotations-bbox.csv
│ └── test-annotations-bbox.csv
└── boxes_Class_names/
└── oidv7-class-descriptions-boxable.csv
```

### 📋 CSV Format

The annotation CSV files should have the following columns:
- `ImageID`: Unique identifier for each image
- `Source`: Source of the image
- `LabelName`: Class identifier
- `Confidence`: Annotation confidence score
- `XMin`, `XMax`, `YMin`, `YMax`: Normalized coordinates [0,1]

## ⚙️ Configuration   

Before running the script, update the following paths in `train_detector.py` to match your data location:
```python
TRAIN_DIR = "/path/to/your/data/target_dir/train"
VALIDATION_DIR = "/path/to/your/data/target_dir/validation"
TEST_DIR = "/path/to/your/data/target_dir/test"
TRAIN_BOXES_CSV = "/path/to/your/data/boxes/oidv6-train-annotations-bbox.csv"
VALIDATION_BOXES_CSV = "/path/to/your/data/boxes/validation-annotations-bbox.csv"
TEST_BOXES_CSV = "/path/to/your/data/boxes/test-annotations-bbox.csv"
CLASS_DESCRIPTIONS_FILE = "/path/to/your/data/boxes_Class_names/oidv7-class-descriptions-boxable.csv"
```

For testing/debugging with smaller datasets:
```python
MAX_TRAIN_ANNOTATIONS = None # Set to a number for smaller dataset
MAX_VAL_ANNOTATIONS = None
MAX_TEST_ANNOTATIONS = None
```

## 🚀 Usage

Run the training pipeline:

```bash
python train_detector.py
```

The script will:
1. Verify CSV file formats
2. Load annotations
3. Extract class names
4. Create and preprocess datasets
5. Build and train the model
6. Evaluate the model
7. Save the model for deployment

## 📊 Output

The pipeline generates several outputs:

- `models/`: Directory containing saved models
  - `best_model.h5`: Best model based on validation accuracy
  - `final_model.h5`: Final model after training
  - `training_history.csv`: Training metrics history

- `logs/`: TensorBoard logs for monitoring training

- `examples/`: Visualization of training examples with bounding boxes

- `predictions/`: Visualization of model predictions on test data

- `deployment_model/`: Deployment-ready models
  - Standard TensorFlow model (SavedModel and H5)
  - TensorFlow Lite models (standard and quantized)
  - TensorRT optimized model (if available)
  - Example inference scripts

## 📊 Visualization Tools

The pipeline includes advanced visualization capabilities:

- Training examples visualization with:
  - Ground truth bounding boxes (green)
  - Class labels with human-readable names
  - Confidence scores
- Prediction visualization with:
  - Predicted bounding boxes (red)
  - Side-by-side comparison of predictions vs. ground truth
  - Prediction confidence scores

## 🧠 Model Architecture

The model uses an EfficientNetB3-based architecture with the following features:

- **Backbone**: EfficientNetB3 pretrained on ImageNet
- **Detection Head**:
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense(1024) with ReLU activation
  - Dropout(0.3)
  - Dense(512) with ReLU activation
  - Dropout(0.2)
- **Outputs**:
  - Classification: Dense(num_classes) with softmax
  - Box regression: Dense(4)

## 📈 Metrics and Evaluation

The model is evaluated using multiple metrics:

- **Classification**:
  - Classification accuracy
  - Sparse categorical crossentropy loss
- **Box Regression**:
  - Mean squared error (MSE)
- **Evaluation Visualization**:
  - Automatic generation of prediction images
  - Visual comparison of predicted vs. ground truth boxes
  - Confidence scores for each prediction

## 🔄 Callbacks and Monitoring

The pipeline includes several callbacks for monitoring and optimization:

- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Stops training if no improvement (patience: 3 epochs)
- **ReduceLROnPlateau**: Reduces learning rate when model plateaus
  - Reduction factor: 0.2
  - Patience: 2 epochs
  - Minimum LR: 0.00001
- **TensorBoard**: Real-time monitoring of:
  - Training metrics
  - Weight histograms
  - Loss graphs

## ⚡ Performance Optimizations

The pipeline includes several optimizations to improve performance:

- **Mixed Precision**: Training with mixed float16/float32 precision
- **Data Prefetching**: Optimized data pipeline using tf.data
- **GPU Memory Management**:
  - Dynamic memory growth
  - Configurable memory limit (default: 10GB)
- **Batch Processing**:
  - Adaptive batch sizing
  - Dynamic padding for bounding boxes

## 🔧 Advanced Configuration

### GPU Memory Management

By default, the script configures GPU with 10GB memory limit. Adjust this based on your hardware:

```python
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=10000)])  # 10GB
```

Also, enable memory growth to avoid TensorFlow reserving all GPU memory at the start:

```python
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Training Parameters

Modify these parameters in the `train_model` function:

```python
epochs = 10  # Number of training epochs
learning_rate = 0.001  # Initial learning rate
batch_size = 16  # Batch size for training
```

## 🔍 Troubleshooting

### Common Issues

1. **GPU Out of Memory**:
   - Reduce batch size
   - Reduce image size
   - Increase memory limit if available

2. **Slow Training**:
   - Ensure GPU is being used correctly
   - Check for data loading bottlenecks
   - Consider using a smaller subset of data for testing

3. **Poor Model Performance**:
   - Increase training epochs
   - Adjust learning rate
   - Add more data augmentation
   - Fine-tune the base model

## 📚 References

- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or support, please open an issue on GitHub or contact [jeronimorepetto@gmail.com](mailto:jeronimorepetto@gmail.com).

