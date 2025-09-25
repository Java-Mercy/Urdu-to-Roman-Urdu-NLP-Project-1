#!/usr/bin/env python3
"""
Deployment script for Urdu to Roman Urdu Translator
Supports both Streamlit and Gradio deployment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_streamlit():
    """Run Streamlit app"""
    print("🚀 Starting Streamlit app...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped.")

def run_gradio():
    """Run Gradio app"""
    print("🚀 Starting Gradio app...")
    try:
        subprocess.run([sys.executable, "gradio_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Gradio app stopped.")

def main():
    print("🌙 Urdu to Roman Urdu Translator - Deployment Script")
    print("=" * 60)
    
    # Check if model files exist
    model_files = [
        "models/best_model_exp_1.pth",
        "models/best_model_exp_2.pth", 
        "models/best_model_exp_3.pth"
    ]
    
    tokenizer_files = [
        "tokenizers/exp_1_tokenizers.pkl",
        "tokenizers/exp_2_tokenizers.pkl",
        "tokenizers/exp_3_tokenizers.pkl"
    ]
    
    missing_files = []
    for file in model_files + tokenizer_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Please ensure all model and tokenizer files are in place.")
        return
    
    print("✅ All required files found!")
    
    # Install requirements
    if not install_requirements():
        return
    
    # Choose deployment option
    print("\n🎯 Choose deployment option:")
    print("1. Gradio (Recommended - Better for ML models)")
    print("2. Streamlit (More customizable)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            run_gradio()
            break
        elif choice == "2":
            run_streamlit()
            break
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
