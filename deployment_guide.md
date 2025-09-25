# 🚀 Deployment Guide - Urdu to Roman Urdu Translator

## 📋 Pre-Deployment Checklist

### ✅ Required Files
- [ ] `streamlit_app.py` - Main application
- [ ] `requirements.txt` - Dependencies
- [ ] `README.md` - Documentation
- [ ] Model files (`.pth` files)
- [ ] Tokenizer files (`.pkl` files)

### 📁 Directory Structure Setup

Create this structure in your project:

```
your-project/
├── streamlit_app.py
├── requirements.txt
├── README.md
├── models/
│   ├── best_model_exp_1.pth
│   ├── best_model_exp_2.pth
│   └── best_model_exp_3.pth
└── tokenizers/
    ├── exp_1_tokenizers.pkl
    ├── exp_2_tokenizers.pkl
    └── exp_3_tokenizers.pkl
```

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)

#### Steps:
1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/urdu-translator.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Configure App**
   - App will be available at: `https://yourusername-urdu-translator-main-xxxxx.streamlit.app`
   - Update model paths if needed

#### Pros:
- ✅ Free
- ✅ Easy deployment
- ✅ Automatic updates
- ✅ Good performance

#### Cons:
- ❌ 1GB file size limit
- ❌ Limited customization

### Option 2: Hugging Face Spaces (FREE)

#### Steps:
1. **Create Hugging Face Account**
   - Go to [huggingface.co](https://huggingface.co)
   - Create account and verify email

2. **Create New Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as SDK
   - Set visibility to "Public"

3. **Upload Files**
   - Upload all your files to the space
   - Ensure `streamlit_app.py` is the main file

4. **Configure Space**
   - Add `requirements.txt` to the space
   - Set hardware to "CPU" (free) or "GPU" (paid)

#### Pros:
- ✅ Free tier available
- ✅ 5GB file size limit
- ✅ Great for ML projects
- ✅ Community features

#### Cons:
- ❌ Slower than Streamlit Cloud
- ❌ Limited free GPU time

### Option 3: Railway (PAID - $5/month)

#### Steps:
1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy from GitHub**
   - Connect your GitHub repository
   - Railway will auto-detect Streamlit
   - Deploy automatically

3. **Configure Environment**
   - Set environment variables if needed
   - Configure build settings

#### Pros:
- ✅ Fast deployment
- ✅ Good performance
- ✅ Easy scaling
- ✅ Custom domains

#### Cons:
- ❌ Paid service
- ❌ More complex setup

## 🔧 Model File Organization

### From Your Colab Training:

1. **Download Model Files**
   ```bash
   # From your Colab, download these files:
   - best_model_exp_1.pth
   - best_model_exp_2.pth  
   - best_model_exp_3.pth
   ```

2. **Download Tokenizer Files**
   ```bash
   # From your Colab, download these files:
   - /content/tokenizer_checkpoints_exp_1/*.pkl
   - /content/tokenizer_checkpoints_exp_2/*.pkl
   - /content/tokenizer_checkpoints_exp_3/*.pkl
   ```

3. **Organize Files**
   ```bash
   mkdir models
   mkdir tokenizers
   
   # Move model files
   mv best_model_exp_*.pth models/
   
   # Move tokenizer files
   mv *tokenizers*.pkl tokenizers/
   ```

## 📝 Environment Configuration

### For Streamlit Cloud:
- No additional configuration needed
- Files are automatically detected

### For Hugging Face Spaces:
- Add `requirements.txt` to the space
- Set hardware type in space settings

### For Railway:
- Create `railway.toml` (optional):
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "streamlit run streamlit_app.py --server.port $PORT"
```

## 🚀 Quick Deployment Commands

### GitHub Setup:
```bash
# Initialize repository
git init
git add .
git commit -m "Add Urdu to Roman Urdu translator"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/urdu-translator.git
git branch -M main
git push -u origin main
```

### Local Testing:
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

## 🔍 Troubleshooting

### Common Issues:

1. **File Not Found Error**
   - Check file paths in `streamlit_app.py`
   - Ensure all files are uploaded

2. **Model Loading Error**
   - Verify model files are not corrupted
   - Check PyTorch version compatibility

3. **Memory Issues**
   - Use smaller model (Experiment 1)
   - Optimize model loading

4. **Slow Loading**
   - Use `@st.cache_resource` (already implemented)
   - Consider model quantization

### File Size Optimization:

1. **Compress Model Files**
   ```python
   # In your training code, save with compression
   torch.save(model.state_dict(), 'model.pth', _use_new_zipfile_serialization=False)
   ```

2. **Use Smaller Models**
   - Experiment 1 is smallest
   - Consider model pruning

## 📊 Performance Tips

### For Production:

1. **Model Optimization**
   - Use `torch.jit.script()` for faster inference
   - Consider model quantization

2. **Caching**
   - Already implemented with `@st.cache_resource`
   - Consider Redis for session caching

3. **Error Handling**
   - Add try-catch blocks
   - Implement fallback mechanisms

## 🎯 Post-Deployment

### Testing:
1. **Test all models** (Experiment 1, 2, 3)
2. **Test example poems**
3. **Test custom input**
4. **Check performance** on different devices

### Monitoring:
1. **Check app logs** for errors
2. **Monitor usage** and performance
3. **Update models** as needed

### Sharing:
1. **Share the public URL**
2. **Update README** with live link
3. **Add to your portfolio**

## 🎉 Success!

Once deployed, your app will be available at:
- **Streamlit Cloud**: `https://yourusername-urdu-translator-main-xxxxx.streamlit.app`
- **Hugging Face**: `https://huggingface.co/spaces/yourusername/urdu-translator`
- **Railway**: `https://yourproject.railway.app`

**Congratulations! Your Urdu to Roman Urdu translator is now live! 🚀**
