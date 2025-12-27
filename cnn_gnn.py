"""IMU Character Recognition Pipeline

This module contains a compact, readable example pipeline to train a
multi-task model on IMU pen data. The script shows an end-to-end flow:

- Load pickled IMU and ground-truth data
- Normalize and pad variable-length sequences
- Extract features with a small CNN trunk
- Model temporal/relational structure with a simple GCN
- Combine into a multi-task model (classification + trajectory regression)

This file is intentionally straightforward to make it easy to read and
adapt. It is not organized as a library; it is a runnable example script.

Notes
- Expected data files: data/all_x_dat_imu.pkl, data/all_gt.pkl
- Keep changes minimal when extending; this file focuses on clarity.
"""

import os
import pickle
import logging
import time
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model, load_model
from keras.layers import (
    Conv1D, Input, Dense, Dropout, Bidirectional, LSTM, GlobalAveragePooling1D,
    RepeatVector, Concatenate
)
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tabulate import tabulate

# Global variables to control behavior
DISPLAY_ALL_RESULTS = False
NUM_EPOCHS = 50
SHOW_NORMALIZED_SAMPLE = False
LOG_FILE = None  # Set to a file path to enable logging to a specific file
DISPLAY_TRAINING_PROGRESS = True

# Configure logging
if LOG_FILE:
    logging.basicConfig(
        filename=LOG_FILE, level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
else:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Helper function to log messages
def log_message(message: str) -> None:
    """Log a message to stdout and the configured logger.

    This keeps the previous behavior (print + logging.info) but makes the
    contract explicit via typing and a short docstring.
    """
    print(message)
    logging.info(message)

# File paths
imu_data_file = 'data/all_x_dat_train_imu.pkl'
gt_data_file  = 'data/all_train_gt.pkl'
traj_data_file= 'data/all_y_dat_train.pkl'

model_file = 'data/mtl_model.keras'

# Load the provided pkl files
start_time = time.time()
log_message("Loading IMU data and ground truth labels...")
# Load IMU data
with open(imu_data_file, 'rb') as f:
    imu_data = pickle.load(f)
log_message(f"IMU data loaded successfully. Number of samples: {len(imu_data)}")

# Load ground truth labels
with open(gt_data_file, 'rb') as f:
    gt_data = pickle.load(f)
log_message(f"Ground truth labels loaded successfully. Number of labels: {len(gt_data)}")

# Load trajectory data
with open(traj_data_file, 'rb') as f:
    y_traj = pickle.load(f)
    y_traj = np.array(y_traj)

log_message(f"Trajectory data loaded. Shape: {y_traj.shape}")

# Check if the number of IMU data samples matches the number of ground truth labels
if len(imu_data) != len(gt_data):
    raise ValueError(
        f"Mismatch in number of samples: IMU data has {len(imu_data)} samples, "
        f"but ground truth has {len(gt_data)} labels."
    )
log_message(f"Step 1: Data Loading completed in {time.time() - start_time:.2f} seconds")

# Convert ground truth labels to categorical format
start_time = time.time()
log_message("Converting ground truth labels to categorical format...")
# Convert labels (A-Z) to numerical format for training
# Les données sont déjà numériques, on les passe directement
gt_data_categorical = to_categorical(gt_data)
log_message(f"Ground truth labels converted successfully. Shape: {gt_data_categorical.shape}")
log_message(f"Step 1: Data Conversion completed in {time.time() - start_time:.2f} seconds")

# Step 1: Data Preprocessing
start_time = time.time()
log_message("Starting data preprocessing...")
log_message("Normalizing IMU data...")
scaler = MinMaxScaler()
# Normalize each sample using MinMaxScaler to bring all features into the same range
imu_data_normalized = [scaler.fit_transform(sample) for sample in imu_data]
if SHOW_NORMALIZED_SAMPLE:
    log_message(
        f"IMU data normalized successfully. Example normalized sample: "
        f"{imu_data_normalized[0]}"
    )

log_message("Padding IMU data sequences...")
# Pad sequences to make them uniform in length for model compatibility
imu_data_padded = pad_sequences(
    imu_data_normalized, padding='post', dtype='float32'
)
log_message(
    f"IMU data padded successfully. Data shape after padding: "
    f"{imu_data_padded.shape}"
)
log_message(f"Step 1: Data Preprocessing completed in {time.time() - start_time:.2f} seconds")

# If trajectory data is shorter than IMU data, pad it
TARGET_LEN = imu_data_padded.shape[1]
if y_traj.shape[1] != TARGET_LEN:
    temp_traj = np.zeros((len(y_traj), TARGET_LEN, 2))
    for i, t in enumerate(y_traj):
        L = min(len(t), TARGET_LEN)
        temp_traj[i, :L, :] = t[:L, :]
    y_traj = temp_traj



# Check if model file exists to avoid retraining
if os.path.exists(model_file):
    log_message("Loading existing MTL model from file...")
    mtl_model = load_model(model_file)
    log_message("Model loaded successfully.")
else:
    # Step 2: Feature Extraction & Sequence Modeling (CNN + LSTM)
    start_time = time.time()
    log_message("Building CNN + LSTM architecture...")
    
    # Input layer
    input_layer = Input(shape=(imu_data_padded.shape[1], 13))
    
    # CNN Layers (padding='same' to preserve sequence length)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)
    
    # LSTM Layer for temporal dependencies
    # return_sequences=True keeps the time dimension (Batch, Steps, Features)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # --- TASK HEADS ---
    
    # 1. Classification Head
    # Collapse the sequence to a single vector for classification
    y_class = GlobalAveragePooling1D()(x)
    classification_output = Dense(
        gt_data_categorical.shape[1], activation='softmax', name='classification'
    )(y_class)
    
    # 2. Autoencoder Head (Sequence-to-Sequence)
    # Reconstructs the input IMU data (13 features) for each timestep
    autoencoder_output = Dense(13, activation='linear', name='autoencoder')(x)
    
    # 3. Trajectory Head (Sequence-to-Sequence) with CONDITIONING
    # We take the classification probabilities (Batch, Classes), repeat them 
    # for each timestep (Batch, Steps, Classes), and merge with the LSTM features.
    # This tells the drawing arm WHAT letter it is supposed to draw.
    
    # Repeat the class probabilities to match the sequence length
    class_context = RepeatVector(imu_data_padded.shape[1])(classification_output)
    
    # Concatenate LSTM features (x) with Class Context
    combined_features = Concatenate()([x, class_context])
    
    # Predict the (x, y) coordinate for each timestep using the combined knowledge
    trajectory_output = Dense(2, activation='linear', name='trajectory')(combined_features)
    
    log_message("Building MTL model with conditioning...")
    # Create the MTL model combining all outputs
    mtl_model = Model(
        inputs=input_layer, 
        outputs=[classification_output, autoencoder_output, trajectory_output]
    )
    log_message("MTL model created successfully.")

    # Compile and train the model
    log_message("Compiling and training the MTL model...")
    # Compile the model with appropriate loss functions for each task
    mtl_model.compile(
        optimizer='adam',
        loss={
            'classification': 'categorical_crossentropy',
            'autoencoder': 'mse',
            'trajectory': 'mae'  # Switch to MAE for sharper reconstruction
        },
        loss_weights={
            'classification': 1.0, 
            'autoencoder': 1.0,
            'trajectory': 50.0  # Drastically increase weight to prioritize drawing
        },
        metrics={'classification': ['accuracy'], 'trajectory': ['mse']}
    )
    log_message("MTL model compiled successfully.")

    if DISPLAY_TRAINING_PROGRESS:
        log_message("Starting model training...")
    # Train the model on the IMU data and ground truth labels
    mtl_model.fit(
        imu_data_padded, [gt_data_categorical, imu_data_padded, y_traj],
        epochs=NUM_EPOCHS, batch_size=32, verbose=DISPLAY_TRAINING_PROGRESS
    )
    log_message("MTL model training completed.")
    log_message(f"Step 4: MTL Model Training completed in {time.time() - start_time:.2f} seconds")

    # Save the trained model to a file
    mtl_model.save(model_file)
    log_message("MTL model saved to file.")

# Step 5: Displaying IMU Character, Ground Truth Character, and Accuracy in Table Format
start_time = time.time()
log_message("Evaluating model and displaying results...")
# Predict character labels from IMU data
predicted_labels, _, _ = mtl_model.predict(imu_data_padded)
# Convert predicted labels to character format
predicted_chars = [chr(np.argmax(pred) + ord('A')) for pred in predicted_labels]
# Ground truth characters
ground_truth_chars = [chr(int(label) + ord('A')) for label in gt_data]

# Evaluate the model on the IMU data and get accuracy
# Evaluate the model on the IMU data and get accuracy
evaluation = mtl_model.evaluate(
    imu_data_padded, [gt_data_categorical, imu_data_padded, y_traj], verbose=0, return_dict=True
)
log_message(f"Evaluation keys: {evaluation.keys()}")
# Try to find the accuracy key dynamically or fallback to classification_accuracy
acc_key = next((k for k in evaluation.keys() if 'accuracy' in k), None)
if acc_key:
    classification_accuracy = evaluation[acc_key]
else:
    log_message("WARNING: Could not find accuracy metric in evaluation results.")
    classification_accuracy = 0.0

# Create a table with IMU character, ground truth character, and accuracy
table_data = []
mismatched_data = []
for i in range(len(predicted_chars)):
    if predicted_chars[i] != ground_truth_chars[i]:
        mismatched_data.append(
            [i + 1, predicted_chars[i], ground_truth_chars[i]]
        )
    if DISPLAY_ALL_RESULTS or predicted_chars[i] != ground_truth_chars[i]:
        table_data.append(
            [i + 1, predicted_chars[i], ground_truth_chars[i]]
        )

# Print the table if DISPLAY_ALL_RESULTS is True
if DISPLAY_ALL_RESULTS:
    log_message(
        tabulate(
            table_data,
            headers=["Sample #", "Predicted Character", "Ground Truth Character"],
            tablefmt="grid"
        )
    )
log_message(f'Classification Accuracy: {classification_accuracy * 100:.2f}%')

# Print mismatched characters
if mismatched_data:
    log_message("\nMismatched Characters:")
    log_message(
        tabulate(
            mismatched_data,
            headers=["Sample #", "Predicted Character", "Ground Truth Character"],
            tablefmt="grid"
        )
    )

    # Post-mortem analysis of mismatched characters
    log_message("\nPost-mortem Analysis of Mismatched Characters:")
    for sample in mismatched_data:
        sample_num, predicted_char, ground_truth_char = sample
        log_message(
            f"Sample {sample_num}: Predicted '{predicted_char}', but expected "
            f"'{ground_truth_char}'. Possible reasons could be model "
            f"overfitting, insufficient training data diversity, or difficulty "
            f"in distinguishing similar characters."
        )
else:
    log_message("\nNo mismatched characters found.")
