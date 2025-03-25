"""
Object Detection Training Pipeline

This module implements a complete pipeline for training an object detection model using TensorFlow.
It handles data loading, preprocessing, model creation, training and evaluation.

Key Features:
- Supports Open Images dataset format
- GPU acceleration with memory management
- Data augmentation and preprocessing
- EfficientNetB3-based detection model
- Mixed precision training
- Visualization tools for debugging

Dependencies:
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

Author: [Your Name]
Date: [Current Date]
"""
 
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm

# GPU Configuration
"""
GPU Memory Management:
- Memory growth is enabled to prevent TensorFlow from allocating all GPU memory at startup
- Memory limit is set to 10GB to allow other processes to use GPU
- A test operation verifies GPU functionality
"""
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

if gpus:
    try:
        # Memory growth must be set before GPUs have been initialized
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
        
        # Memory limit configuration
        # Note: 10GB is chosen as a balance between model requirements and system resources
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=10000)])  # 10GB
        print("GPU configured with 10GB of memory")
        
        # GPU verification test
        # Performs matrix multiplication to ensure GPU is working correctly
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("Test operation on GPU:", c)
    except RuntimeError as e:
        print("Error configuring GPU:", e)
else:
    print("No GPUs detected, using CPU")

# Data Directory Configuration
"""
Directory Structure:
/media/username/Storage/AI-Boost/
├── target_dir/
│   ├── train/
│   ├── validation/
│   └── test/
├── boxes/
│   ├── oidv6-train-annotations-bbox.csv
│   ├── validation-annotations-bbox.csv
│   └── test-annotations-bbox.csv
└── boxes_Class_names/
    └── oidv7-class-descriptions-boxable.csv
"""

TRAIN_DIR = "/media/mturco/Storage/AI-Boost/target_dir/train"
VALIDATION_DIR = "/media/mturco/Storage/AI-Boost/target_dir/validation"
TEST_DIR = "/media/mturco/Storage/AI-Boost/target_dir/test"

# Annotation Files Configuration
"""
CSV Format:
- ImageID: Unique identifier for each image
- Source: Source of the image
- LabelName: Class identifier
- Confidence: Annotation confidence score
- XMin, XMax, YMin, YMax: Normalized coordinates [0,1]
"""

TRAIN_BOXES_CSV = "/media/mturco/Storage/AI-Boost/boxes/oidv6-train-annotations-bbox.csv"
VALIDATION_BOXES_CSV = "/media/mturco/Storage/AI-Boost/boxes/validation-annotations-bbox.csv"
TEST_BOXES_CSV = "/media/mturco/Storage/AI-Boost/boxes/test-annotations-bbox.csv"
CLASS_DESCRIPTIONS_FILE = "/media/mturco/Storage/AI-Boost/boxes_Class_names/oidv7-class-descriptions-boxable.csv"
# Dataset Size Configuration
"""
Control variables for dataset size:
- None values use full dataset
- Can be set to smaller values for testing/debugging
"""
MAX_TRAIN_ANNOTATIONS = None
MAX_VAL_ANNOTATIONS = None
MAX_TEST_ANNOTATIONS = None

def load_annotations(csv_path, max_rows=None):
    """
    Loads and parses object detection annotations from a CSV file.
    
    Technical Details:
    - Uses pandas for efficient CSV reading
    - Supports partial loading for large files
    - Measures loading time for performance monitoring
    
    CSV Structure Expected:
    - ImageID: str - Unique image identifier
    - LabelName: str - Class identifier
    - XMin, YMin, XMax, YMax: float - Normalized coordinates
    
    Args:
        csv_path (str): Full path to the CSV annotation file
        max_rows (int, optional): Maximum number of rows to load, useful for testing
        
    Returns:
        pandas.DataFrame: DataFrame containing the parsed annotations
        
    Performance:
    - Memory usage scales linearly with number of rows
    - Loading time depends on file size and system I/O
    """
    print(f"Loading annotations from {csv_path}...")
    start_time = time.time()
    df = pd.read_csv(csv_path, nrows=max_rows)
    print(f"Loading completed in {time.time() - start_time:.2f} seconds. {len(df)} rows loaded.")
    return df

def check_csv_format(csv_path, max_rows=5):
    """
    Verifies the format of the CSV file and displays the first rows.
    
    Args:
        csv_path (str): Path to the CSV file
        max_rows (int, optional): Number of rows to display
        
    Returns:
        bool: True if the file was read successfully, False otherwise
    """
    try:
        df = pd.read_csv(csv_path, nrows=max_rows)
        print(f"\nColumns in {os.path.basename(csv_path)}:")
        print(df.columns.tolist())
        print(f"\nFirst {max_rows} rows:")
        print(df.head(max_rows))
        return True
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return False

def extract_class_names_from_annotations(train_df, val_df, test_df):
    """
    Extracts unique class names from annotation files.
    
    Args:
        train_df (DataFrame): Training annotations
        val_df (DataFrame): Validation annotations
        test_df (DataFrame): Test annotations
        
    Returns:
        list: List of unique class names
    """
    print("Extracting class names...")
    start_time = time.time()
    
    # Combine all labels
    all_labels = pd.concat([
        train_df['LabelName'],
        val_df['LabelName'],
        test_df['LabelName']
    ]).unique().tolist()
    
    print(f"Extraction completed in {time.time() - start_time:.2f} seconds. {len(all_labels)} classes found.")
    return all_labels

def convert_open_images_to_standard_format(df):
    """
    Converts Open Images format to standard format.
    
    Args:
        df (DataFrame): DataFrame in Open Images format
        
    Returns:
        DataFrame: DataFrame in standard format with columns:
                  filename, class, xmin, ymin, xmax, ymax
    """
    print(f"Converting {len(df)} annotations to standard format...")
    start_time = time.time()
    
    # Create a new DataFrame with standard format
    standard_df = pd.DataFrame()
    
    # Copy necessary columns
    standard_df['filename'] = df['ImageID'] + '.jpg'
    standard_df['class'] = df['LabelName']
    
    # Coordinates in Open Images are already normalized between 0 and 1
    standard_df['xmin'] = df['XMin']
    standard_df['ymin'] = df['YMin']
    standard_df['xmax'] = df['XMax']
    standard_df['ymax'] = df['YMax']
    
    print(f"Conversion completed in {time.time() - start_time:.2f} seconds.")
    return standard_df

def create_annotations_dict(df, class_names):
    """
    Creates an annotations dictionary from a DataFrame.
    
    Args:
        df (DataFrame): DataFrame with annotations in standard format
        class_names (list): List of class names
        
    Returns:
        tuple: (annotations_dict, label_encoder)
            - annotations_dict: Dictionary mapping filenames to lists of (box, label) tuples
            - label_encoder: LabelEncoder for converting class names to indices
    """
    print(f"Creating annotations dictionary for {len(df)} rows...")
    start_time = time.time()
    
    # Create a label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    # Create a dictionary to store annotations by image
    annotations_dict = {}
    
    # Iterate over DataFrame rows
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing annotations")):
        filename = row['filename']
        class_name = row['class']
        
        # Encode the label
        label = label_encoder.transform([class_name])[0]
        
        # Extract box coordinates
        box = [row['ymin'], row['xmin'], row['ymax'], row['xmax']]  # Format [y_min, x_min, y_max, x_max]
        
        # Add annotation to dictionary
        if filename not in annotations_dict:
            annotations_dict[filename] = []
        
        annotations_dict[filename].append((box, label))
    
    print(f"Dictionary created in {time.time() - start_time:.2f} seconds. {len(annotations_dict)} images with annotations.")
    return annotations_dict, label_encoder

def create_dataset(image_dir, annotations_dict):
    """
    Creates a TensorFlow dataset from an image directory and annotations dictionary.
    
    Args:
        image_dir (str): Directory containing images
        annotations_dict (dict): Dictionary mapping filenames to annotations
        
    Returns:
        tf.data.Dataset or None: Dataset containing (filename, boxes, labels) tuples,
                                 or None if an error occurred
    """
    print(f"Creating dataset from {image_dir}...")
    start_time = time.time()
    
    filenames = []
    boxes_list = []
    labels_list = []
    
    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist.")
        return None
    
    # Iterate over annotations
    for filename, annotations in annotations_dict.items():
        # Build full image path
        img_path = os.path.join(image_dir, filename)
        
        # Check if image exists
        if not os.path.exists(img_path):
            continue
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        for box, label in annotations:
            boxes.append(box)
            labels.append(label)
        
        # Add to lists
        filenames.append(img_path)
        
        # Convert to NumPy arrays with padding
        max_boxes = 100  # Maximum number of boxes per image
        padded_boxes = np.zeros((max_boxes, 4), dtype=np.float32)
        padded_labels = np.zeros(max_boxes, dtype=np.int32)
        
        # Fill with actual values
        num_boxes = min(len(boxes), max_boxes)
        padded_boxes[:num_boxes] = np.array(boxes[:num_boxes], dtype=np.float32)
        padded_labels[:num_boxes] = np.array(labels[:num_boxes], dtype=np.int32)
        
        boxes_list.append(padded_boxes)
        labels_list.append(padded_labels)
    
    if not filenames:
        print(f"Error: No images found in {image_dir} that match the annotations.")
        return None
    
    # Convert lists to tensors
    boxes_tensor = np.array(boxes_list)
    labels_tensor = np.array(labels_list)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((filenames, boxes_tensor, labels_tensor))
    
    print(f"Dataset created in {time.time() - start_time:.2f} seconds. {len(filenames)} images included.")
    return dataset

def load_image(img_path, boxes, labels):
    """
    Loads an image and its annotations.
    
    Args:
        img_path (str): Path to the image file
        boxes (tensor): Bounding boxes tensor
        labels (tensor): Labels tensor
        
    Returns:
        tuple: (image, valid_boxes, valid_labels)
            - image: Decoded image tensor
            - valid_boxes: Tensor containing only valid boxes (non-padding)
            - valid_labels: Tensor containing only valid labels (non-padding)
    """
    # Read image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Find valid boxes and labels (non-padding)
    valid_indices = tf.where(tf.reduce_sum(boxes, axis=1) > 0)
    valid_boxes = tf.gather(boxes, valid_indices)
    valid_labels = tf.gather(labels, valid_indices)
    
    # Flatten indices
    valid_boxes = tf.reshape(valid_boxes, [-1, 4])
    valid_labels = tf.reshape(valid_labels, [-1])
    
    return img, valid_boxes, valid_labels

def preprocess_dataset(dataset, img_size, batch_size, augment=False):
    """
    Preprocesses a dataset for training.
    
    Args:
        dataset (tf.data.Dataset): Dataset to preprocess
        img_size (tuple): Target image size (height, width)
        batch_size (int): Batch size
        augment (bool): Whether to apply data augmentation
        
    Returns:
        tf.data.Dataset: Preprocessed dataset
    """
    # Definir la función de preprocesamiento con los parámetros explícitos
    @tf.function
    def preprocess_fn(img_path, boxes, labels):
        # Load image and annotations
        img, valid_boxes, valid_labels = load_image(img_path, boxes, labels)
        
        # Get original dimensions
        original_height = tf.cast(tf.shape(img)[0], tf.float32)
        original_width = tf.cast(tf.shape(img)[1], tf.float32)
        
        # Resize image
        img = tf.image.resize(img, img_size)
        img = img / 255.0  # Normalize to [0,1]
        
        # Data augmentation (if enabled)
        if augment:
            # Random horizontal flip
            flip = tf.random.uniform([], 0, 1) > 0.5
            flip_cond = tf.cast(flip, tf.bool)
            
            # Usar tf.cond en lugar de if para operaciones condicionales en el grafo
            img = tf.cond(
                flip_cond,
                lambda: tf.image.flip_left_right(img),
                lambda: img
            )
            
            # Adjust boxes for flipping usando tf.cond
            valid_boxes = tf.cond(
                flip_cond,
                lambda: tf.stack([
                    valid_boxes[:, 0],           # y_min doesn't change
                    1.0 - valid_boxes[:, 3],     # x_min = 1 - original x_max
                    valid_boxes[:, 2],           # y_max doesn't change
                    1.0 - valid_boxes[:, 1]      # x_max = 1 - original x_min
                ], axis=1),
                lambda: valid_boxes
            )
            
            # Random brightness, contrast, saturation adjustment
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_saturation(img, 0.8, 1.2)
            
            # Ensure values are in [0,1]
            img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img, valid_boxes, valid_labels
    
    # Apply preprocessing
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Function to filter examples without valid boxes
    def has_valid_boxes(img, boxes, labels):
        return tf.greater(tf.shape(boxes)[0], 0)
    
    # Filter examples without valid boxes
    dataset = dataset.filter(has_valid_boxes)
    
    # Batch processing
    # Use padded_batch to handle different numbers of boxes
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            img_size + (3,),                # Image shape
            [None, 4],                      # Variable shape for boxes
            [None]                          # Variable shape for labels
        ),
        padding_values=(
            tf.constant(0.0, dtype=tf.float32),  # Padding for images
            tf.constant(0.0, dtype=tf.float32),  # Padding for boxes
            tf.constant(0, dtype=tf.int32)       # Padding for labels
        )
    )
    
    # Optimize performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def verify_dataset(dataset, class_names):
    """
    Verifies that a dataset has the correct format.
    
    Args:
        dataset (tf.data.Dataset): Dataset to verify
        class_names (list): List of class names
        
    Returns:
        bool: True if verification passed, False otherwise
    """
    try:
        # Get a batch
        for images, boxes, labels in dataset.take(1):
            print(f"Image shape: {images.shape}")
            print(f"Boxes shape: {boxes.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Check image range
            print(f"Image value range: [{tf.reduce_min(images)}, {tf.reduce_max(images)}]")
            
            # Check box range
            print(f"Box value range: [{tf.reduce_min(boxes)}, {tf.reduce_max(boxes)}]")
            
            # Check labels
            unique_labels = tf.unique(tf.reshape(labels, [-1]))[0]
            print(f"Unique labels in batch: {unique_labels.numpy()}")
            
            # Check that labels are within valid range
            max_label = tf.reduce_max(labels)
            if max_label >= len(class_names):
                print(f"Warning! Maximum label ({max_label}) outside class range ({len(class_names)}).")
            
            return True
    except Exception as e:
        print(f"Error verifying dataset: {e}")
        return False

def visualize_examples(dataset, class_names, num_examples=5, img_size=(416, 416), class_descriptions=None):
    """
    Visualizes examples from the dataset with their bounding boxes.
    
    Args:
        dataset (tf.data.Dataset): Dataset to visualize
        class_names (list): List of class names
        num_examples (int): Number of examples to visualize
        img_size (tuple): Image size (height, width)
        class_descriptions (dict): Dictionary mapping class IDs to human-readable names
    """
    # Create directory to save examples
    examples_dir = "examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # Colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    # Take some examples - shuffle first to get different examples each time
    examples = list(dataset.unbatch().shuffle(buffer_size=1000).take(num_examples))
    
    for i, (image, boxes, labels) in enumerate(examples):
        # Convert to NumPy
        image_np = image.numpy()
        boxes_np = boxes.numpy()
        labels_np = labels.numpy()
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_np)
        
        # Draw boxes
        for box, label in zip(boxes_np, labels_np):
            # Get coordinates
            y_min, x_min, y_max, x_max = box
            
            # Convert to pixels
            x_min_px = x_min * img_size[1]
            y_min_px = y_min * img_size[0]
            width_px = (x_max - x_min) * img_size[1]
            height_px = (y_max - y_min) * img_size[0]
            
            # Get color for class
            color = colors[label % len(colors)]
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x_min_px, y_min_px), width_px, height_px,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label - use human-readable name if available
            class_id = class_names[label] if label < len(class_names) else f"Unknown ({label})"
            display_name = class_id
            
            if class_descriptions and class_id in class_descriptions:
                display_name = class_descriptions[class_id]
            
            plt.text(
                x_min_px, y_min_px - 5,
                display_name,
                bbox=dict(facecolor=color, alpha=0.5),
                fontsize=8, color='white'
            )
        
        # Save figure
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(examples_dir, f"example_{i+1}.png"))
        plt.close()
    
    print(f"Examples saved in directory '{examples_dir}'")

def load_class_descriptions(file_path):
    """
    Loads class descriptions from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file with class descriptions
        
    Returns:
        dict: Dictionary mapping class IDs to descriptions
    """
    descriptions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(',')
            if len(line) >= 2:
                class_id = line[0]
                # Join the rest in case there are commas in the description
                description = ','.join(line[1:])
                descriptions[class_id] = description
    return descriptions

def save_model_for_deployment(model, class_names, label_encoder, output_dir='deployment_model'):
    """
    Saves the model and necessary metadata for cloud deployment.
    Includes optimizations with TensorRT and TensorFlow Lite.
    
    Args:
        model (tf.keras.Model): Trained model
        class_names (list): List of class names
        label_encoder (LabelEncoder): Label encoder
        output_dir (str): Directory to save files
    """
    print("\nPreparing model for deployment...")
    start_time = time.time()
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model in SavedModel format (recommended for TF Serving)
    saved_model_path = os.path.join(output_dir, 'saved_model')
    tf.saved_model.save(model, saved_model_path)
    
    # Save model in H5 format (alternative)
    model.save(os.path.join(output_dir, 'model.h5'))
    
    # Optimize with TensorFlow Lite
    print("Optimizing model with TensorFlow Lite...")
    tflite_dir = os.path.join(output_dir, 'tflite')
    os.makedirs(tflite_dir, exist_ok=True)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(os.path.join(tflite_dir, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    # Optimize with quantization to reduce size
    print("Applying quantization to optimize size...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    
    # Save quantized TFLite model
    with open(os.path.join(tflite_dir, 'model_quantized.tflite'), 'wb') as f:
        f.write(tflite_quant_model)
    
    # Optimize with TensorRT if available
    tensorrt_dir = os.path.join(output_dir, 'tensorrt')
    os.makedirs(tensorrt_dir, exist_ok=True)
    
    try:
        print("Attempting to optimize with TensorRT...")
        # Check if TensorRT is available
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
        
        # Conversion parameters
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        conversion_params = conversion_params._replace(
            max_workspace_size_bytes=(1 << 30),  # 1GB
            precision_mode="FP16",  # Use FP16 precision for better performance
            maximum_cached_engines=1
        )
        
        # Convert to TensorRT
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=saved_model_path,
            conversion_params=conversion_params
        )
        converter.convert()
        
        # Save TensorRT optimized model
        trt_saved_model_path = os.path.join(tensorrt_dir, 'saved_model')
        converter.save(trt_saved_model_path)
        print("TensorRT optimized model saved successfully.")
    except (ImportError, tf.errors.NotFoundError) as e:
        print(f"Could not optimize with TensorRT: {e}")
        print("Make sure TensorRT is properly installed.")
    
    # Save metadata necessary for inference
    np.save(os.path.join(output_dir, 'class_names.npy'), np.array(class_names))
    
    # Save label_encoder
    import pickle
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Create a configuration file with important information
    config = {
        'input_shape': (416, 416, 3),
        'num_classes': len(class_names),
        'version': '1.0',
        'date_created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'optimized_versions': {
            'tensorflow': True,
            'tflite': True,
            'tflite_quantized': True,
            'tensorrt': 'tensorrt' in os.listdir(output_dir)
        }
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        import json
        json.dump(config, f, indent=4)
    
    # Create example scripts for inference with different versions
    # Script for standard TensorFlow
    tf_inference_script = """
import tensorflow as tf
import numpy as np
import pickle
import json
import os

def load_deployment_model(model_dir):
    # Load model
    model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'))
    
    # Load metadata
    class_names = np.load(os.path.join(model_dir, 'class_names.npy'), allow_pickle=True)
    
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    return model, class_names, label_encoder, config

def preprocess_image(image_path, target_size=(416, 416)):
    # Load and preprocess image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return tf.expand_dims(img, 0)  # Add batch dimension

def detect_objects(model, image, class_names, confidence_threshold=0.5):
    # Make prediction
    pred_labels, pred_boxes = model.predict(image)
    
    # Get class with highest confidence
    pred_class_idx = np.argmax(pred_labels[0])
    confidence = pred_labels[0][pred_class_idx]
    
    if confidence < confidence_threshold:
        return None, None, 0
    
    # Get class name
    class_name = class_names[pred_class_idx]
    
    # Get box coordinates
    box = pred_boxes[0]  # [y_min, x_min, y_max, x_max]
    
    return class_name, box, confidence

# Example usage
if __name__ == "__main__":
    model_dir = "deployment_model"
    image_path = "test_image.jpg"
    
    # Load model and metadata
    model, class_names, label_encoder, config = load_deployment_model(model_dir)
    
    # Preprocess image
    image = preprocess_image(image_path)
    
    # Detect objects
    class_name, box, confidence = detect_objects(model, image, class_names)
    
    if class_name is not None:
        print(f"Object detected: {class_name}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Box: {box}")
    else:
        print("No objects detected with sufficient confidence.")
"""
    
    # Script for TensorFlow Lite
    tflite_inference_script = """
import tensorflow as tf
import numpy as np
import pickle
import json
import os
import time

def load_tflite_model(model_path):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def preprocess_image(image_path, input_details):
    # Get input size from model
    input_shape = input_details[0]['shape']
    target_size = (input_shape[1], input_shape[2])
    
    # Load and preprocess image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, 0)  # Add batch dimension
    
    return img.numpy()

def detect_objects_tflite(interpreter, input_details, output_details, image, class_names, confidence_threshold=0.5):
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Get results
    pred_labels = interpreter.get_tensor(output_details[0]['index'])
    pred_boxes = interpreter.get_tensor(output_details[1]['index'])
    
    # Get class with highest confidence
    pred_class_idx = np.argmax(pred_labels[0])
    confidence = pred_labels[0][pred_class_idx]
    
    if confidence < confidence_threshold:
        return None, None, 0, inference_time
    
    # Get class name
    class_name = class_names[pred_class_idx]
    
    # Get box coordinates
    box = pred_boxes[0]  # [y_min, x_min, y_max, x_max]
    
    return class_name, box, confidence, inference_time

# Example usage
if __name__ == "__main__":
    model_dir = "deployment_model"
    tflite_model_path = os.path.join(model_dir, "tflite", "model.tflite")
    # You can also use the quantized model:
    # tflite_model_path = os.path.join(model_dir, "tflite", "model_quantized.tflite")
    image_path = "test_image.jpg"
    
    # Load metadata
    class_names = np.load(os.path.join(model_dir, 'class_names.npy'), allow_pickle=True)
    
    # Load TFLite model
    interpreter, input_details, output_details = load_tflite_model(tflite_model_path)
    
    # Preprocess image
    image = preprocess_image(image_path, input_details)
    
    # Detect objects
    class_name, box, confidence, inference_time = detect_objects_tflite(
        interpreter, input_details, output_details, image, class_names)
    
    if class_name is not None:
        print(f"Object detected: {class_name}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Box: {box}")
        print(f"Inference time: {inference_time*1000:.2f} ms")
    else:
        print("No objects detected with sufficient confidence.")
"""

    # Script for TensorRT
    tensorrt_inference_script = """
import tensorflow as tf
import numpy as np
import pickle
import json
import os
import time

def load_tensorrt_model(saved_model_dir):
    # Load TensorRT optimized model
    model = tf.saved_model.load(saved_model_dir)
    
    # Get inference function
    infer = model.signatures['serving_default']
    
    return model, infer

def preprocess_image(image_path, target_size=(416, 416)):
    # Load and preprocess image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return tf.expand_dims(img, 0)  # Add batch dimension

def detect_objects_tensorrt(infer_function, image, class_names, confidence_threshold=0.5):
    # Make prediction
    start_time = time.time()
    results = infer_function(tf.constant(image))
    inference_time = time.time() - start_time
    
    # Extract results
    pred_labels = results['classification'].numpy()
    pred_boxes = results['regression'].numpy()
    
    # Get class with highest confidence
    pred_class_idx = np.argmax(pred_labels[0])
    confidence = pred_labels[0][pred_class_idx]
    
    if confidence < confidence_threshold:
        return None, None, 0, inference_time
    
    # Get class name
    class_name = class_names[pred_class_idx]
    
    # Get box coordinates
    box = pred_boxes[0]  # [y_min, x_min, y_max, x_max]
    
    return class_name, box, confidence, inference_time

# Example usage
if __name__ == "__main__":
    model_dir = "deployment_model"
    tensorrt_model_dir = os.path.join(model_dir, "tensorrt", "saved_model")
    image_path = "test_image.jpg"
    
    # Check if TensorRT model exists
    if not os.path.exists(tensorrt_model_dir):
        print("TensorRT model not found. Make sure it was generated correctly.")
        exit(1)
    
    # Load metadata
    class_names = np.load(os.path.join(model_dir, 'class_names.npy'), allow_pickle=True)
    
    # Load TensorRT model
    model, infer = load_tensorrt_model(tensorrt_model_dir)
    
    # Preprocess image
    image = preprocess_image(image_path)
    
    # Detect objects
    class_name, box, confidence, inference_time = detect_objects_tensorrt(
        infer, image, class_names)
    
    if class_name is not None:
        print(f"Object detected: {class_name}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Box: {box}")
        print(f"Inference time: {inference_time*1000:.2f} ms")
    else:
        print("No objects detected with sufficient confidence.")
"""

    # Save scripts
    with open(os.path.join(output_dir, 'inference_tf.py'), 'w') as f:
        f.write(tf_inference_script)
    
    with open(os.path.join(output_dir, 'inference_tflite.py'), 'w') as f:
        f.write(tflite_inference_script)
    
    with open(os.path.join(output_dir, 'inference_tensorrt.py'), 'w') as f:
        f.write(tensorrt_inference_script)
    
    # Create a README with instructions
    readme_content = """# Object Detection Model

This directory contains the trained and optimized object detection model ready for deployment.

## Available Versions

- **Standard TensorFlow**: Complete model in SavedModel and H5 formats.
- **TensorFlow Lite**: Optimized version for mobile devices and edge computing.
- **Quantized TensorFlow Lite**: Smaller and more efficient version.
- **TensorRT** (if available): Optimized version for NVIDIA GPUs.

## Directory Structure

- `saved_model/`: Model in TensorFlow SavedModel format
- `model.h5`: Model in Keras H5 format
- `tflite/`: Models optimized with TensorFlow Lite
  - `model.tflite`: Standard TFLite model
  - `model_quantized.tflite`: Quantized TFLite model
- `tensorrt/` (if available): Model optimized with TensorRT
- `class_names.npy`: Class names that the model can detect
- `label_encoder.pkl`: Label encoder
- `config.json`: Model configuration

## Example Scripts

- `inference_tf.py`: Example inference with standard TensorFlow
- `inference_tflite.py`: Example inference with TensorFlow Lite
- `inference_tensorrt.py`: Example inference with TensorRT

## Requirements

- TensorFlow 2.x
- NumPy
- Pillow (for image processing)

## Cloud Usage

This model is prepared to be deployed on services such as:

- Google Cloud AI Platform
- AWS SageMaker
- Azure Machine Learning
- TensorFlow Serving

## Performance Comparison

| Version | Size | Inference Time | Accuracy |
|---------|------|----------------|----------|
| TensorFlow | Large | Baseline | High |
| TFLite | Medium | Faster | Medium-High |
| TFLite quantized | Small | Faster | Medium |
| TensorRT | Large | Very Fast | High |

Choose the version that best fits your deployment needs.
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"Model prepared for deployment in {output_dir} in {time.time() - start_time:.2f} seconds.")
    print(f"Generated files:")
    for file in os.listdir(output_dir):
        print(f" - {file}")

def main():
    """
    Main function to run the training pipeline.
    """
    print("Verifying CSV file format:")
    # Check if CSV files exist and have the correct format
    train_csv_valid = check_csv_format(TRAIN_BOXES_CSV)
    val_csv_valid = check_csv_format(VALIDATION_BOXES_CSV)
    test_csv_valid = check_csv_format(TEST_BOXES_CSV)
    
    if not (train_csv_valid and val_csv_valid and test_csv_valid):
        print("Error: One or more CSV files are invalid or missing.")
        return None, None, None, None, None
    
    # Load annotations
    train_df = load_annotations(TRAIN_BOXES_CSV, max_rows=MAX_TRAIN_ANNOTATIONS)
    val_df = load_annotations(VALIDATION_BOXES_CSV, max_rows=MAX_VAL_ANNOTATIONS)
    test_df = load_annotations(TEST_BOXES_CSV, max_rows=MAX_TEST_ANNOTATIONS)
    
    # Check if directories exist
    if not (os.path.exists(TRAIN_DIR) and os.path.exists(VALIDATION_DIR) and os.path.exists(TEST_DIR)):
        print(f"Error: One or more directories do not exist:")
        print(f"  Train: {TRAIN_DIR} - Exists: {os.path.exists(TRAIN_DIR)}")
        print(f"  Validation: {VALIDATION_DIR} - Exists: {os.path.exists(VALIDATION_DIR)}")
        print(f"  Test: {TEST_DIR} - Exists: {os.path.exists(TEST_DIR)}")
        return None, None, None, None, None
    
    # Extract class names
    class_names = extract_class_names_from_annotations(train_df, val_df, test_df)
    
    # Convert to standard format
    train_df_std = convert_open_images_to_standard_format(train_df)
    val_df_std = convert_open_images_to_standard_format(val_df)
    test_df_std = convert_open_images_to_standard_format(test_df)
    
    # Create annotations dictionaries
    train_annotations_dict, label_encoder = create_annotations_dict(train_df_std, class_names)
    val_annotations_dict, _ = create_annotations_dict(val_df_std, class_names)
    test_annotations_dict, _ = create_annotations_dict(test_df_std, class_names)
    
    # Create datasets
    train_dataset = create_dataset(TRAIN_DIR, train_annotations_dict)
    val_dataset = create_dataset(VALIDATION_DIR, val_annotations_dict)
    test_dataset = create_dataset(TEST_DIR, test_annotations_dict)
    
    # Check if datasets were created successfully
    if train_dataset is None or val_dataset is None or test_dataset is None:
        print("Error: Failed to create one or more datasets.")
        return None, None, None, None, None
    
    # Preprocess datasets
    img_size = (416, 416)
    batch_size = 16
    
    train_dataset = preprocess_dataset(train_dataset, img_size, batch_size, augment=True)
    val_dataset = preprocess_dataset(val_dataset, img_size, batch_size, augment=False)
    test_dataset = preprocess_dataset(test_dataset, img_size, batch_size, augment=False)
    
    # Verify datasets
    print("Verifying datasets:")
    train_verified = verify_dataset(train_dataset, class_names)
    val_verified = verify_dataset(val_dataset, class_names)
    test_verified = verify_dataset(test_dataset, class_names)
    
    if not (train_verified and val_verified and test_verified):
        print("Error: Dataset verification failed.")
        return None, None, None, None, None
    
    # Visualize examples
    try:
        class_descriptions = load_class_descriptions(CLASS_DESCRIPTIONS_FILE)
        visualize_examples(train_dataset, class_names, num_examples=5, 
                          img_size=img_size, class_descriptions=class_descriptions)
    except Exception as e:
        print(f"Warning: Could not visualize examples: {e}")
    
    # Build and train model
    model = build_model(len(class_names))
    history = train_model(model, train_dataset, val_dataset, epochs=10)
    
    # Load class descriptions for evaluation
    try:
        class_descriptions = load_class_descriptions(CLASS_DESCRIPTIONS_FILE)
        readable_class_names = [class_descriptions.get(class_id, class_id) for class_id in class_names]
    except Exception as e:
        print(f"Warning: Could not load class descriptions: {e}")
        readable_class_names = class_names
    
    # Evaluate model
    evaluate_model(model, test_dataset, class_names, readable_class_names)
    
    # Save model for deployment
    save_model_for_deployment(model, class_names, label_encoder)
    
    return train_dataset, val_dataset, test_dataset, class_names, label_encoder

def build_model(num_classes, input_shape=(416, 416, 3)):
    """
    Builds an object detection model based on EfficientNetB3.
    
    Args:
        num_classes (int): Number of classes to detect
        input_shape (tuple): Input shape (height, width, channels)
        
    Returns:
        tf.keras.Model: Compiled model for object detection
    """
    print("\nBuilding detection model...")
    start_time = time.time()
    
    # Use EfficientNetB3 as base model
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add detection layers
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output for classification
    classification_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    # Output for box regression
    regression_output = tf.keras.layers.Dense(4, name='regression')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=[classification_output, regression_output])
    
    print(f"Model built in {time.time() - start_time:.2f} seconds.")
    return model

def train_model(model, train_dataset, val_dataset, epochs=10):
    """
    Trains the model with the provided datasets.
    
    Args:
        model (tf.keras.Model): Model to train
        train_dataset (tf.data.Dataset): Training dataset
        val_dataset (tf.data.Dataset): Validation dataset
        epochs (int): Number of training epochs
        
    Returns:
        tf.keras.callbacks.History: Training history
    """
    print("\nStarting model training...")
    start_time = time.time()
    
    # Use mixed precision to speed up training and reduce memory usage
    mixed_precision_policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)
    print("Using mixed precision for training")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'classification': 'sparse_categorical_crossentropy',
            'regression': 'mse'
        },
        metrics={
            'classification': 'accuracy',
            'regression': 'mse'
        }
    )
    
    # Create directory to save models
    os.makedirs('models', exist_ok=True)
    
    # Callbacks for training
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True,
            monitor='val_classification_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True,
            monitor='val_classification_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.2,
            patience=2,
            min_lr=0.00001,
            monitor='val_classification_accuracy'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('models/final_model.h5')
    
    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('models/training_history.csv', index=False)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes).")
    
    return history

def evaluate_model(model, test_dataset, class_names, readable_class_names):
    """
    Evaluates the model on the test set and displays prediction examples.
    
    Args:
        model (tf.keras.Model): Trained model
        test_dataset (tf.data.Dataset): Test dataset
        class_names (list): List of class IDs
        readable_class_names (list): List of human-readable class names
    """
    print("\nEvaluating model on test set...")
    start_time = time.time()
    
    # Evaluate model
    results = model.evaluate(test_dataset)
    
    # Show results
    print("\nEvaluation results:")
    metric_names = ['loss', 'classification_loss', 'regression_loss', 
                   'classification_accuracy', 'regression_mse']
    
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")
    
    # Visualize some predictions
    print("\nVisualizing predictions on test set...")
    
    # Create directory to save examples
    os.makedirs('predictions', exist_ok=True)
    
    # Get a batch from test set
    for images, true_boxes, true_labels in test_dataset.take(1):
        # Make predictions
        pred_labels, pred_boxes = model.predict(images)
        
        # Visualize first 5 images
        for i in range(min(5, len(images))):
            plt.figure(figsize=(10, 10))
            
            # Show image
            img = images[i].numpy()
            plt.imshow(img)
            
            # Get true boxes and labels
            valid_indices = tf.where(tf.reduce_sum(true_boxes[i], axis=1) > 0)
            valid_boxes = tf.gather(true_boxes[i], valid_indices).numpy()
            valid_labels = tf.gather(true_labels[i], valid_indices).numpy()
            
            # Draw true boxes (in green)
            for box, label in zip(valid_boxes, valid_labels):
                y_min, x_min, y_max, x_max = box
                width = x_max - x_min
                height = y_max - y_min
                
                # Create rectangle
                rect = patches.Rectangle(
                    (x_min * 416, y_min * 416),
                    width * 416,
                    height * 416,
                    linewidth=2,
                    edgecolor='g',
                    facecolor='none'
                )
                plt.gca().add_patch(rect)
                
                # Add label
                class_id = class_names[int(label)]
                class_name = readable_class_names[int(label)]
                plt.text(
                    x_min * 416,
                    y_min * 416 - 5,
                    f"True: {class_name}",
                    color='g',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7)
                )
            
            # Get predicted label with highest confidence
            pred_label = np.argmax(pred_labels[i])
            pred_box = pred_boxes[i]
            
            # Draw predicted box (in red)
            y_min, x_min, y_max, x_max = pred_box
            width = x_max - x_min
            height = y_max - y_min
            
            # Create rectangle
            rect = patches.Rectangle(
                (x_min * 416, y_min * 416),
                width * 416,
                height * 416,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            plt.gca().add_patch(rect)
            
            # Add label
            class_id = class_names[pred_label]
            class_name = readable_class_names[pred_label]
            confidence = pred_labels[i][pred_label]
            plt.text(
                x_min * 416,
                y_min * 416 - 20,
                f"Pred: {class_name} ({confidence:.2f})",
                color='r',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7)
            )
            
            plt.title(f"Prediction {i+1}")
            plt.axis('off')
            
            # Save image
            plt.savefig(f'predictions/prediction_{i+1}.png')
            plt.close()
    
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, class_names, label_encoder = main()

