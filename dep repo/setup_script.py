#!/usr/bin/env python3
"""
Setup script for CNN-LSTM IDS deployment
This script helps you organize your files for deployment
"""

import os
import shutil
import sys

def create_deployment_structure():
    """Create the proper directory structure for deployment"""
    
    print("🚀 Setting up deployment structure...")
    
    # Create directories
    directories = [
        "deployment",
        "deployment/models",
        "deployment/data",
        "deployment/notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Files to create/copy
    files_to_create = {
        "deployment/requirements.txt": """streamlit==1.28.0
tensorflow==2.13.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
imbalanced-learn==0.11.0
pickle-mixin==1.0.2""",
        
        "deployment/README.md": """# CNN-LSTM Network Intrusion Detection System

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (if you haven't already)
   ```bash
   python train_model.py
   ```

3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

## File Structure
- `app.py` - Main Streamlit application
- `train_model.py` - Training script (cleaned from Colab)
- `models/` - Saved model and preprocessors
- `requirements.txt` - Python dependencies

## Model Files Needed
Make sure these files are in the `models/` directory:
- `cnn_lstm_unsw_nb15_model.h5` - Trained model
- `encoder.pkl` - OneHot encoder for categorical features
- `scaler.pkl` - Standard scaler for numerical features  
- `feature_info.pkl` - Feature metadata

## Dataset
Place your `NF-UNSW-NB15-v2.csv` file in the `data/` directory.
""",

        "deployment/.streamlit/config.toml": """[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200
""",

        "deployment/run_local.py": """#!/usr/bin/env python3
import subprocess
import sys
import os

def check_requirements():
    try:
        import streamlit
        import tensorflow
        import sklearn
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_model_files():
    required_files = [
        "models/cnn_lstm_unsw_nb15_model.h5",
        "models/encoder.pkl", 
        "models/scaler.pkl",
        "models/feature_info.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\\nPlease run train_model.py first to generate these files.")
        return False
    else:
        print("✅ All model files found")
        return True

def main():
    print("🚀 Starting CNN-LSTM IDS Application")
    print("=" * 40)
    
    if not check_requirements():
        sys.exit(1)
    
    if not check_model_files():
        print("\\n💡 To train the model, run: python train_model.py")
        sys.exit(1)
    
    print("\\n🌐 Starting Streamlit application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
""",

        "deployment/Dockerfile": """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create models directory
RUN mkdir -p models

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
""",

        "deployment/.gitignore": """# Model files (too large for git)
models/*.h5
models/*.pkl

# Data files
data/*.csv
data/*.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Streamlit
.streamlit/secrets.toml

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
"""
    }
    
    # Create .streamlit directory
    os.makedirs("deployment/.streamlit", exist_ok=True)
    
    # Create all files
    for file_path, content in files_to_create.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✅ Created file: {file_path}")

def copy_existing_files():
    """Copy existing files to deployment directory"""
    print("\n📁 Checking for existing files to copy...")
    
    # Files to look for and copy
    file_mappings = {
        "copy_cnn_lstm_(unsw_nb15).py": "deployment/notebooks/original_notebook.py",
        "NF-UNSW-NB15-v2.csv": "deployment/data/NF-UNSW-NB15-v2.csv",
        "cnn_lstm_UNSW-NB15_model.h5": "deployment/models/cnn_lstm_unsw_nb15_model.h5",
        "encoder.pkl": "deployment/models/encoder.pkl",
        "scaler.pkl": "deployment/models/scaler.pkl",
        "feature_info.pkl": "deployment/models/feature_info.pkl"
    }
    
    copied_files = []
    missing_files = []
    
    for source, destination in file_mappings.items():
        if os.path.exists(source):
            try:
                shutil.copy2(source, destination)
                copied_files.append(f"{source} → {destination}")
                print(f"✅ Copied: {source} → {destination}")
            except Exception as e:
                print(f"❌ Error copying {source}: {e}")
        else:
            missing_files.append(source)
    
    if missing_files:
        print(f"\n⚠️  Missing files (you'll need to provide these):")
        for file in missing_files:
            print(f"   - {file}")
    
    return copied_files, missing_files

def print_next_steps(copied_files, missing_files):
    """Print instructions for next steps"""
    print("\n" + "=" * 50)
    print("🎉 DEPLOYMENT SETUP COMPLETE!")
    print("=" * 50)
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Files copied: {len(copied_files)}")
    print(f"   ⚠️  Files missing: {len(missing_files)}")
    
    print(f"\n📂 Your deployment structure:")
    print("""
deployment/
├── app.py                 # Main Streamlit app
├── train_model.py         # Clean training script  
├── run_local.py          # Local development runner
├── requirements.txt       # Dependencies
├── Dockerfile            # For containerization
├── README.md             # Documentation
├── models/               # Model files
│   ├── cnn_lstm_unsw_nb15_model.h5
│   ├── encoder.pkl
│   ├── scaler.pkl
│   └── feature_info.pkl
├── data/                 # Dataset files
│   └── NF-UNSW-NB15-v2.csv
└── notebooks/            # Original code
    └── original_notebook.py
    """)
    
    print(f"\n🚀 Next Steps:")
    
    if missing_files:
        print(f"\n1️⃣ TRAIN YOUR MODEL (if you haven't already):")
        print(f"   cd deployment")
        print(f"   python train_model.py")
        print(f"   # This will generate the missing .pkl files")
    
    print(f"\n2️⃣ INSTALL DEPENDENCIES:")
    print(f"   cd deployment")
    print(f"   pip install -r requirements.txt")
    
    print(f"\n3️⃣ RUN LOCALLY:")
    print(f"   python run_local.py")
    print(f"   # OR")
    print(f"   streamlit run app.py")
    
    print(f"\n4️⃣ DEPLOY ONLINE:")
    print(f"   Option A: Streamlit Cloud")
    print(f"   - Push to GitHub")
    print(f"   - Connect at share.streamlit.io")
    print(f"   ")
    print(f"   Option B: Docker")
    print(f"   - docker build -t ids-app .")
    print(f"   - docker run -p 8501:8501 ids-app")
    
    if missing_files and 'NF-UNSW-NB15-v2.csv' in missing_files:
        print(f"\n⚠️  IMPORTANT: Download the UNSW-NB15 dataset")
        print(f"   Place NF-UNSW-NB15-v2.csv in deployment/data/")

def main():
    print("🛠️  CNN-LSTM IDS Deployment Setup")
    print("=" * 40)
    
    try:
        # Create directory structure and files
        create_deployment_structure()
        
        # Copy existing files
        copied_files, missing_files = copy_existing_files()
        
        # Print next steps
        print_next_steps(copied_files, missing_files)
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()