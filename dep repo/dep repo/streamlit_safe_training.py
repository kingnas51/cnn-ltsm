# Streamlit-safe training script for CNN-LSTM IDS model
# This version avoids plotting issues in cloud deployment

import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score

# Only import plotting libraries when needed
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    st.warning("⚠️ Plotting libraries not available. Training will continue without visualizations.")

# Configuration
MODEL_SAVE_PATH = "models"
DATA_PATH = "NF-UNSW-NB15-v2.csv"

def ensure_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

def preprocess_unsw(df):
    """Preprocessing function for UNSW-NB15 dataset"""
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    st.write(f"Dataset columns: {len(df.columns)}")

    # Create binary label (normal=0, attack=1)
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 0 else 1)

    # Remove infinite values and NaNs
    original_shape = df.shape
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    st.write(f"Data cleaned: {original_shape} → {df.shape}")

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
    st.write(f"Using {len(available_features)} of {len(initial_features)} selected features")

    # Define categorical and numerical features
    cat_features = ['PROTOCOL']
    num_features = [f for f in available_features if f not in cat_features]

    st.write(f"Categorical features: {len(cat_features)}")
    st.write(f"Numerical features: {len(num_features)}")

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
        ensure_directory(base_path)
        
        # Save the OneHotEncoder
        encoder_path = os.path.join(base_path, 'encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
        
        # Save the StandardScaler
        scaler_path = os.path.join(base_path, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
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
        
        st.success("✅ Preprocessors saved successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error saving preprocessors: {str(e)}")
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

def train_model_streamlit():
    """Streamlit interface for model training"""
    st.title("🎯 CNN-LSTM Model Training")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload UNSW-NB15 Dataset (CSV)", 
        type=['csv'],
        help="Upload the NF-UNSW-NB15-v2.csv file"
    )
    
    if uploaded_file is not None:
        # Load dataset
        try:
            with st.spinner("Loading dataset..."):
                nb15_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Dataset loaded: {nb15_df.shape}")
            
            # Show dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", f"{nb15_df.shape[0]:,}")
            with col2:
                st.metric("Features", nb15_df.shape[1])
            with col3:
                if 'Label' in nb15_df.columns:
                    attack_rate = nb15_df['Label'].mean()
                    st.metric("Attack Rate", f"{attack_rate:.1%}")
            
            # Training configuration
            st.subheader("🔧 Training Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
                batch_size = st.selectbox("Batch Size", [128, 256, 512], index=1)
            
            with col2:
                epochs = st.slider("Max Epochs", 5, 50, 15)
                validation_split = st.slider("Validation Split", 0.1, 0.3, 0.15, 0.05)
            
            # Start training button
            if st.button("🚀 Start Training", type="primary"):
                train_model_pipeline(nb15_df, test_size, batch_size, epochs, validation_split)
                
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
    
    else:
        st.info("👆 Please upload the UNSW-NB15 dataset to start training")
        
        # Show sample data format
        with st.expander("📋 Expected Data Format"):
            st.write("""
            The CSV file should contain columns like:
            - PROTOCOL, IN_BYTES, IN_PKTS, OUT_BYTES, OUT_PKTS
            - FLOW_DURATION_MILLISECONDS, MIN_TTL, MAX_TTL
            - Label (0 for normal, 1+ for attacks)
            """)

def train_model_pipeline(nb15_df, test_size, batch_size, epochs, validation_split):
    """Execute the complete training pipeline"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Preprocessing
        status_text.text("🔄 Preprocessing data...")
        progress_bar.progress(10)
        
        X_unsw, y_unsw, unsw_encoder, unsw_scaler, num_features = preprocess_unsw(nb15_df)
        
        # Step 2: Save preprocessors
        status_text.text("💾 Saving preprocessors...")
        progress_bar.progress(20)
        
        save_preprocessors(unsw_encoder, unsw_scaler, num_features)
        
        # Step 3: Split data
        status_text.text("🔀 Splitting data...")
        progress_bar.progress(30)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_unsw, y_unsw, test_size=test_size, stratify=y_unsw, random_state=42
        )
        
        # Step 4: Balance data
        status_text.text("⚖️ Balancing data with SMOTE...")
        progress_bar.progress(40)
        
        sampler = SMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        
        # Step 5: Reshape for CNN-LSTM
        status_text.text("🔄 Reshaping for CNN-LSTM...")
        progress_bar.progress(50)
        
        X_train_3d = X_res.reshape((X_res.shape[0], X_res.shape[1], 1))
        X_test_3d = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Step 6: Create model
        status_text.text("🏗️ Creating model...")
        progress_bar.progress(60)
        
        input_shape = (X_train_3d.shape[1],)
        model = create_model(input_shape)
        
        # Step 7: Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        # Step 8: Train model
        status_text.text("🎯 Training model...")
        progress_bar.progress(70)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Training with progress updates
        history = model.fit(
            X_train_3d, y_res,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            class_weight=class_weight_dict,
            callbacks=[early_stopping],
            verbose=0  # Suppress verbose output for Streamlit
        )
        
        progress_bar.progress(90)
        
        # Step 9: Save model
        status_text.text("💾 Saving model...")
        ensure_directory(MODEL_SAVE_PATH)
        model_path = os.path.join(MODEL_SAVE_PATH, "cnn_lstm_unsw_nb15_model.h5")
        model.save(model_path)
        
        # Step 10: Evaluate
        status_text.text("📊 Evaluating model...")
        y_pred_proba = model.predict(X_test_3d, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Save results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'training_history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            }
        }
        
        results_path = os.path.join(MODEL_SAVE_PATH, 'results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        progress_bar.progress(100)
        status_text.text("✅ Training completed!")
        
        # Display results
        st.success("🎉 Model Training Completed Successfully!")
        
        # Show metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        with col5:
            st.metric("ROC-AUC", f"{roc_auc:.4f}")
        
        # Training history (only if plotting is available)
        if PLOTTING_AVAILABLE:
            st.subheader("📈 Training History")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(history.history['accuracy'], label='Training Accuracy')
                ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax.set_title('Model Accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(history.history['loss'], label='Training Loss')
                ax.plot(history.history['val_loss'], label='Validation Loss')
                ax.set_title('Model Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)
        
        # Files created
        st.subheader("📁 Files Created")
        created_files = [
            f"✅ {model_path}",
            f"✅ {MODEL_SAVE_PATH}/encoder.pkl",
            f"✅ {MODEL_SAVE_PATH}/scaler.pkl", 
            f"✅ {MODEL_SAVE_PATH}/feature_info.pkl",
            f"✅ {results_path}"
        ]
        
        for file in created_files:
            st.write(file)
            
        st.info("🚀 You can now use these files in your main Streamlit app!")
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.write("**Error details:**", str(e))

if __name__ == "__main__":
    train_model_streamlit()