# 🌙 Urdu to Roman Urdu Neural Machine Translation

A sophisticated Neural Machine Translation system that translates Urdu poetry to Roman Urdu using PyTorch and Streamlit.

## 🚀 Features

- **Neural Machine Translation**: 2-layer BiLSTM encoder + 4-layer LSTM decoder
- **Custom BPE Tokenization**: Built from scratch without external libraries
- **Multiple Model Experiments**: 3 different hyperparameter configurations
- **Interactive Web Interface**: Beautiful Streamlit app for easy translation
- **Real-time Translation**: Instant Urdu to Roman Urdu conversion
- **Confidence Scoring**: Model confidence for each translation

## 📊 Model Performance

| Experiment | Model Size | Embed Dim | Hidden Dim | Encoder Layers | Decoder Layers | Accuracy |
|------------|------------|-----------|------------|----------------|----------------|----------|
| Experiment 1 | Small | 128 | 256 | 1 | 2 | ~73% |
| Experiment 2 | Medium | 256 | 512 | 2 | 3 | ~75% |
| Experiment 3 | Large | 512 | 512 | 3 | 4 | ~74% |

## 🛠️ Installation

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/Java-Mercy/Urdu-to-Roman-Urdu-NLP-Project-1.git
cd Urdu-to-Roman-Urdu-NLP-Project-1
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app (Choose one)**

**Option A: Gradio (Recommended)**
```bash
python gradio_app.py
```

**Option B: Streamlit**
```bash
streamlit run streamlit_app.py
```

**Option C: Use deployment script**
```bash
python deploy.py
```

### Cloud Deployment

#### Option 1: Streamlit Cloud (Recommended)

1. **Upload to GitHub**
   - Push your code to a GitHub repository
   - Include all model files and tokenizers

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Deploy automatically

#### Option 2: Hugging Face Spaces

1. **Create a new Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Create a new Streamlit space

2. **Upload files**
   - Upload your code and model files
   - Set the app file to `streamlit_app.py`

## 📁 Project Structure

```
urdu-roman-translator/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── models/                  # Trained model files
│   ├── best_model_exp_1.pth
│   ├── best_model_exp_2.pth
│   └── best_model_exp_3.pth
├── tokenizers/              # Tokenizer files
│   ├── exp_1_tokenizers.pkl
│   ├── exp_2_tokenizers.pkl
│   └── exp_3_tokenizers.pkl
└── my-model-final-collab.ipynb  # Training notebook
```

## 🎯 Usage

### Web Interface

1. **Open the app** in your browser
2. **Select a model** from the sidebar (Experiment 2 recommended)
3. **Click "Load Model"** to initialize the selected model
4. **Enter Urdu text** in the input area
5. **Click "Translate"** to get Roman Urdu translation
6. **View results** with confidence scores

### Example Translations

| Urdu | Roman Urdu |
|------|------------|
| یہ نہ تھی ہماری قسمت کہ وصال یار ہوتا | ye na thi hamaari qismat ke visaal yaar hota |
| اگر اپنا کہا آپ ہی سمجھتے تو کیا کہتے | agar apna kaha aap hi samajhte to kya kahte |
| دل سے جو بات نکلتی ہے اثر رکھتی ہے | dil se jo baat nikalati hai asar rakhti hai |

## 🔧 Technical Details

### Model Architecture

- **Encoder**: 2-layer Bidirectional LSTM
- **Decoder**: 4-layer LSTM with Attention Mechanism
- **Embedding**: Custom word embeddings
- **Tokenization**: Byte-Pair Encoding (BPE) from scratch

### Training Configuration

- **Dataset**: Urdu Ghazals from Rekhta
- **Training Split**: 50% train, 25% validation, 25% test
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Cross-entropy with teacher forcing
- **Evaluation Metrics**: BLEU, Perplexity, Character Error Rate

## 📈 Performance Metrics

- **BLEU Score**: 0.00-0.15 (typical for low-resource languages)
- **Perplexity**: 1.15-1.25
- **Character Error Rate**: 0.15-0.25
- **Word Accuracy**: 73-75%

## 🚀 Deployment Options

### Free Platforms

1. **Streamlit Cloud** - Easiest deployment
2. **Hugging Face Spaces** - Great for ML projects
3. **Railway** - Modern platform with good free tier

### Paid Platforms

1. **Heroku** - Reliable but requires credit card
2. **AWS/GCP/Azure** - Full control but more complex

## 🔍 Troubleshooting

### Common Issues

1. **Model not loading**
   - Check file paths in `streamlit_app.py`
   - Ensure model files are in correct directory

2. **Translation not working**
   - Verify tokenizer files are present
   - Check if model is properly loaded

3. **Memory issues**
   - Use smaller model (Experiment 1)
   - Reduce batch size in deployment

### File Size Limits

- **Streamlit Cloud**: 1GB total
- **Hugging Face**: 5GB per file
- **Heroku**: 500MB slug size

## 📝 License

This project is for educational purposes. Please respect the original dataset sources and cite appropriately.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the training notebook for technical details

## 🎉 Acknowledgments

- **Dataset**: Urdu Ghazals from Rekhta
- **Framework**: PyTorch for deep learning
- **Interface**: Streamlit for web deployment
- **Evaluation**: NLTK and SacreBLEU for metrics

---

**Built with ❤️ for preserving Urdu poetry through technology**