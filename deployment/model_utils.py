import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import os

class IDSModelLoader:
    """Utility class for loading and managing the IDS model components"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        self.model_path = model_path
        
    def load_model(self, model_path):
        """Load the trained CNN-LSTM model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def load_preprocessors(self, scaler_path, encoder_path):
        """Load the preprocessing components"""
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)
            
            print("✅ Preprocessors loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading preprocessors: {str(e)}")
            return False
    
    def preprocess_single_sample(self, sample_data):
        """Preprocess a single network flow sample"""
        try:
            # Define the expected feature columns based on your training data
            expected_features = [
                'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
                'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL',
                'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN',
                'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES'
            ]
            
            # Convert input to DataFrame if it's not already
            if not isinstance(sample_data, pd.DataFrame):
                if isinstance(sample_data, dict):
                    sample_data = pd.DataFrame([sample_data])
                else:
                    # Assume it's a list or array
                    sample_data = pd.DataFrame([sample_data], columns=expected_features[:len(sample_data)])
            
            # Separate categorical and numerical features
            cat_features = ['PROTOCOL']  # Adjust based on your actual categorical features
            num_features = [f for f in sample_data.columns if f not in cat_features]
            
            # Process categorical features
            if self.encoder and cat_features:
                encoded_cats = self.encoder.transform(sample_data[cat_features])
            else:
                encoded_cats = np.array([]).reshape(1, 0)
            
            # Process numerical features
            if self.scaler and num_features:
                scaled_nums = self.scaler.transform(sample_data[num_features])
            else:
                scaled_nums = sample_data[num_features].values
            
            # Combine features
            if encoded_cats.shape[1] > 0:
                X = np.concatenate([encoded_cats, scaled_nums], axis=1)
            else:
                X = scaled_nums
            
            # Reshape for CNN-LSTM (samples, features, 1)
            X_3d = X.reshape((X.shape[0], X.shape[1], 1))
            
            return X_3d
            
        except Exception as e:
            print(f"❌ Error preprocessing sample: {str(e)}")
            return None
    
    def predict(self, processed_data):
        """Make prediction on preprocessed data"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Get prediction probabilities
            prediction_proba = self.model.predict(processed_data, verbose=0)
            
            # Convert to binary prediction
            prediction = (prediction_proba > 0.5).astype(int)
            
            # Calculate confidence
            confidence = float(prediction_proba[0][0] if prediction[0][0] == 1 else 1 - prediction_proba[0][0])
            
            # Determine threat level
            if confidence > 0.8:
                threat_level = "HIGH"
            elif confidence > 0.6:
                threat_level = "MEDIUM"
            else:
                threat_level = "LOW"
            
            return {
                'prediction': int(prediction[0][0]),
                'probability': float(prediction_proba[0][0]),
                'confidence': confidence,
                'threat_level': threat_level,
                'class_name': 'Attack' if prediction[0][0] == 1 else 'Normal'
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None
    
    def predict_batch(self, data_batch):
        """Make predictions on a batch of data"""
        try:
            predictions = []
            
            for i, sample in enumerate(data_batch):
                # Preprocess individual sample
                processed_sample = self.preprocess_single_sample(sample)
                
                if processed_sample is not None:
                    # Make prediction
                    result = self.predict(processed_sample)
                    if result:
                        result['sample_id'] = i
                        predictions.append(result)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error making batch predictions: {str(e)}")
            return []

def save_preprocessors(scaler, encoder, scaler_path='scaler.pkl', encoder_path='encoder.pkl'):
    """Save preprocessing components"""
    try:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
        
        print(f"✅ Preprocessors saved: {scaler_path}, {encoder_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving preprocessors: {str(e)}")
        return False

def create_sample_data():
    """Create sample network flow data for testing"""
    
    sample_flows = [
        {
            'PROTOCOL': 6,  # TCP
            'IN_BYTES': 2048,
            'IN_PKTS': 15,
            'OUT_BYTES': 1024,
            'OUT_PKTS': 10,
            'FLOW_DURATION_MILLISECONDS': 5000,
            'MIN_TTL': 64,
            'MAX_TTL': 64,
            'LONGEST_FLOW_PKT': 1500,
            'SHORTEST_FLOW_PKT': 64,
            'label': 'Normal'
        },
        {
            'PROTOCOL': 17,  # UDP
            'IN_BYTES': 50000,
            'IN_PKTS': 1000,
            'OUT_BYTES': 100,
            'OUT_PKTS': 2,
            'FLOW_DURATION_MILLISECONDS': 100,
            'MIN_TTL': 1,
            'MAX_TTL': 255,
            'LONGEST_FLOW_PKT': 50,
            'SHORTEST_FLOW_PKT': 50,
            'label': 'Attack'
        },
        {
            'PROTOCOL': 6,  # TCP
            'IN_BYTES': 1500,
            'IN_PKTS': 8,
            'OUT_BYTES': 800,
            'OUT_PKTS': 6,
            'FLOW_DURATION_MILLISECONDS': 3000,
            'MIN_TTL': 64,
            'MAX_TTL': 64,
            'LONGEST_FLOW_PKT': 1200,
            'SHORTEST_FLOW_PKT': 100,
            'label': 'Normal'
        }
    ]
    
    return pd.DataFrame(sample_flows)