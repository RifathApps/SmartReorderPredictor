"""
Setup script for Smart Reorder Predictor
This script initializes the project and installs dependencies.
"""

import os
import sys
import subprocess

def create_directories():
    """Create necessary directories."""
    dirs = ['data', 'models', 'notebooks', 'logs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def install_dependencies():
    """Install Python dependencies."""
    print("\nInstalling dependencies...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    print("✓ Dependencies installed successfully")

def main():
    """Main setup function."""
    print("=" * 60)
    print("Smart Reorder Predictor - Project Setup")
    print("=" * 60)
    
    try:
        create_directories()
        install_dependencies()
        
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Download Rossmann dataset from Kaggle")
        print("2. Place train.csv and store.csv in the 'data/' directory")
        print("3. Run: streamlit run app.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
