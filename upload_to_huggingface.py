#!/usr/bin/env python3
"""
Script to upload your trained models to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_models_to_hf():
    """Upload models and tokenizers to Hugging Face"""
    
    # Your Hugging Face repository
    repo_id = "tahir-next/Urdu-RomanUrdu"
    
    # Initialize HF API
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository {repo_id} is ready")
    except Exception as e:
        print(f"⚠️ Repository creation: {e}")
    
    # Files to upload
    files_to_upload = {
        # Model files
        "models/best_model_exp_1.pth": "best_model_exp_1.pth",
        "models/best_model_exp_2.pth": "best_model_exp_2.pth", 
        "models/best_model_exp_3.pth": "best_model_exp_3.pth",
        
        # Tokenizer files
        "tokenizers/exp_1_tokenizers.pkl": "exp_1_tokenizers.pkl",
        "tokenizers/exp_2_tokenizers.pkl": "exp_2_tokenizers.pkl",
        "tokenizers/exp_3_tokenizers.pkl": "exp_3_tokenizers.pkl",
    }
    
    # Upload files
    for local_path, remote_filename in files_to_upload.items():
        if os.path.exists(local_path):
            print(f"📤 Uploading {local_path} as {remote_filename}...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_filename,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"✅ Successfully uploaded {remote_filename}")
            except Exception as e:
                print(f"❌ Error uploading {remote_filename}: {e}")
        else:
            print(f"⚠️ File not found: {local_path}")
    
    print(f"\n🎉 Upload complete! Your models are available at:")
    print(f"https://huggingface.co/{repo_id}")

def upload_folder_to_hf():
    """Alternative: Upload entire folders"""
    
    repo_id = "tahir-next/Urdu-RomanUrdu"
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    # Upload models folder
    if os.path.exists("models"):
        print("📤 Uploading models folder...")
        api.upload_folder(
            folder_path="models",
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="models"
        )
        print("✅ Models folder uploaded")
    
    # Upload tokenizers folder
    if os.path.exists("tokenizers"):
        print("📤 Uploading tokenizers folder...")
        api.upload_folder(
            folder_path="tokenizers",
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="tokenizers"
        )
        print("✅ Tokenizers folder uploaded")

if __name__ == "__main__":
    print("🚀 Uploading Urdu to Roman Urdu models to Hugging Face...")
    print("=" * 60)
    
    # Check if HF_TOKEN is set
    if not os.getenv("HF_TOKEN"):
        print("❌ Please set your HF_TOKEN environment variable:")
        print("export HF_TOKEN=your_huggingface_token")
        exit(1)
    
    print("Choose upload method:")
    print("1. Upload individual files")
    print("2. Upload folders")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        upload_models_to_hf()
    elif choice == "2":
        upload_folder_to_hf()
    else:
        print("Invalid choice. Exiting.")
