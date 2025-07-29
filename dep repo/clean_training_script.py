# Clean training script for CNN-LSTM IDS model
# Remove all Colab-specific commands

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Reshape, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Configuration
MODEL_SAVE_PATH = "models"
DATA_PATH = "NF-UNSW-NB15-v2.csv"  # Update this path to your dataset

# Create models directory
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def preprocess_unsw(df):
    """Preprocessing function for UNSW-NB15 dataset"""
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    print("Columns after stripping whitespace:", df.columns.tolist())

    # Create binary label (normal=0, attack=1)
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 0 else 1)

    # Remove infinite values and NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Feature selection - Use features actually available in the dataframe
    available_cols = df.columns.tolist()
    initial_features = [
        'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
        'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL',
        'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN',
        'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
        'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES',
        'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
        'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES',
        'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN',
        'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID',
        'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE'
    ]

    available_features = [f for f in initial_features if f in available_cols]
    print(f"Using {len(available_features)} of {len(initial_features)} selected features")
    print("Available selected features:", available_features)

    # Define categorical and numerical features
    cat_features = ['PROTOCOL']
    num_features = [f for f in available_features if f not in cat_features]

    print("Categorical features:", cat_features)
    print("Numerical features:", num_features)

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[cat_features])

    # Scale numerical features
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(df[num_features])

    # Combine features
    X = np.concatenate([encoded_cats, scaled_nums], axis=1)
    y = df['Label'].values

    return X, y, encoder, scaler, num_features

def save_preprocessors(encoder, scaler, num_features, base_path=MODEL_SAVE_PATH):
    """Save preprocessing components for deployment"""
    try:
        # Save the OneHotEncoder
        encoder_path = os.path.join(base_path, 'encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
        print(f"✅ Encoder saved to: {encoder_path}")
        
        # Save the StandardScaler
        scaler_path = os.path.join(base_path, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved to: {scaler_path}")
        
        # Save the feature names for reference
        feature_info = {
            'num_features': num_features,
            'cat_features': ['PROTOCOL'],
            'feature_order': list(encoder.get_feature_names_out()) + num_features,
            'total_features': len(encoder.get_feature_names_out()) + len(num_features)
        }
        
        feature_path = os.path.join(base_path, 'feature_info.pkl')
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"✅ Feature info saved to: {feature_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving preprocessors: {str(e)}")
        return False

def create_model(input_shape):
    """Create CNN-LSTM model architecture"""
    model = Sequential(name="CNN_LSTM_IDS")

    # Input reshaping for time-series like data
    model.add(Reshape((input_shape[0], 1), input_shape=input_shape))

    # CNN Feature Extraction
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # LSTM Temporal Analysis
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dropout(0.3))

    # Classification Head
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def main():
    """Main training function"""
    print("🚀 Starting CNN-LSTM IDS Training")
    print("=" * 50)
    
    # Load dataset
    try:
        print(f"📂 Loading dataset from: {DATA_PATH}")
        nb15_df = pd.read_csv(DATA_PATH)
        print(f"Dataset shape: {nb15_df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Dataset file '{DATA_PATH}' not found!")
        print("Please update the DATA_PATH variable with the correct path to your dataset.")
        return
    
    # Preprocessing
    print("\n🔄 Preprocessing data...")
    X_unsw, y_unsw, unsw_encoder, unsw_scaler, num_features = preprocess_unsw(nb15_df)
    
    # Save preprocessors
    print("\n💾 Saving preprocessors...")
    save_preprocessors(unsw_encoder, unsw_scaler, num_features)
    
    # Split and balance data
    print("\n🔀 Splitting and balancing data...")
    X_train_unsw, X_test_unsw, y_train_unsw, y_test_unsw = train_test_split(
        X_unsw, y_unsw, test_size=0.3, stratify=y_unsw, random_state=42
    )

    # Resample training data using SMOTE
    sampler = SMOTE(random_state=42)
    X_res_unsw, y_res_unsw = sampler.fit_resample(X_train_unsw, y_train_unsw)
    print(f"Resampled training data shape: {X_res_unsw.shape}")

    # Reshape for CNN-LSTM
    X_train_unsw_3d = X_res_unsw.reshape((X_res_unsw.shape[0], X_res_unsw.shape[1], 1))
    X_test_unsw_3d = X_test_unsw.reshape((X_test_unsw.shape[0], X_test_unsw.shape[1], 1))
    print(f"3D Training shape: {X_train_unsw_3d.shape}")
    print(f"3D Testing shape: {X_test_unsw_3d.shape}")

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_unsw), y=y_train_unsw)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Create model
    print("\n🏗️ Creating model...")
    input_shape = (X_train_unsw_3d.shape[1],)
    model = create_model(input_shape)
    model.summary()
    
    # Training parameters
    batch_size = 256
    epochs = 15
    validation_split = 0.15

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train the model
    print("\n🎯 Starting model training...")
    history = model.fit(
        X_train_unsw_3d,
        y_res_unsw,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1
    )

    # Save model
    model_path = os.path.join(MODEL_SAVE_PATH, "cnn_lstm_unsw_nb15_model.h5")
    model.save(model_path)
    print(f"✅ Model saved as: {model_path}")
    
    # Evaluation
    print("\n📊 Evaluating model...")
    y_pred_proba = model.predict(X_test_unsw_3d)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_unsw, y_pred)
    precision = precision_score(y_test_unsw, y_pred)
    recall = recall_score(y_test_unsw, y_pred)
    f1 = f1_score(y_test_unsw, y_pred)
    roc_auc = roc_auc_score(y_test_unsw, y_pred_proba)

    # Print results
    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC AUC:     {roc_auc:.4f}")
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    results_path = os.path.join(MODEL_SAVE_PATH, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✅ Results saved to: {results_path}")
    print("\n🎉 Training completed successfully!")
    
    return model, history, results

if __name__ == "__main__":
    model, history, results = main()