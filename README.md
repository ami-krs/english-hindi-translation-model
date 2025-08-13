# 🌐 English to Hindi Translation Model

A **Neural Machine Translation (NMT)** model that translates English text to Hindi using **Bidirectional LSTM with Attention Mechanism**. This model is specifically optimized for **CPU training** on MacBook Air and similar devices.

## ✨ Features

- **🔄 Bidirectional LSTM Encoder**: Reads text both forward and backward for better understanding
- **🎯 Bahdanau Attention**: Focuses on relevant parts of input during translation
- **⚡ CPU Optimized**: Balanced architecture for reasonable training time (15-20 minutes)
- **🚀 Early Stopping**: Automatically stops training when loss stops improving
- **📊 Learning Rate Scheduling**: Reduces learning rate for better convergence
- **🔍 Beam Search**: Advanced decoding for higher quality translations
- **💾 Model Persistence**: Saves best and final model weights

## 🏗️ Architecture

```
Input Text → Embedding → Bidirectional LSTM → Attention → Decoder LSTM → Output Text
```

- **Encoder**: Bidirectional LSTM (256 units) with Dropout regularization
- **Attention**: Bahdanau attention mechanism for context-aware translation
- **Decoder**: LSTM (512 units) with attention integration
- **Embedding**: 128-dimensional word vectors

## 📁 Project Structure

```
Language_Translatio_Model/
├── translate_eng2hindi.py    # Main training and inference script
├── Dataset_English_Hindi.csv # Training dataset (English-Hindi pairs)
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd Language_Translatio_Model
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python translate_eng2hindi.py
```

## 📊 Training Details

- **Dataset Size**: 1,500 English-Hindi sentence pairs
- **Training Time**: 15-20 minutes on MacBook Air CPU
- **Epochs**: Up to 50 (with early stopping)
- **Batch Size**: 16
- **Expected Final Loss**: < 1.0 for good accuracy

## 🎯 Usage

### Training
The script automatically:
1. Loads and preprocesses the dataset
2. Trains the encoder-decoder model
3. Saves the best model weights
4. Tests translations on sample sentences

### Interactive Translation
After training, you can:
- Translate individual sentences
- Use interactive mode for continuous translation
- Compare beam search vs greedy decoding

### Sample Translations
```python
# Test sentences included in the script:
"hello" → "नमस्ते"
"thank you" → "धन्यवाद"
"good night" → "शुभ रात्रि"
"i love you" → "मैं तुमसे प्यार करता हूं"
```

## 🔧 Customization

### Model Parameters
```python
embedding_dim = 128    # Word embedding dimensions
units = 256           # LSTM units
batch_size = 16       # Training batch size
epochs = 50           # Maximum training epochs
```

### Training Data
- Modify `df = df_small.iloc[:1500]` to change dataset size
- Adjust `patience = 8` for early stopping sensitivity
- Change `learning_rate` in optimizer for different convergence

## 📈 Performance Optimization

### For Faster Training
- Reduce `embedding_dim` and `units`
- Decrease dataset size
- Use smaller `batch_size`

### For Better Accuracy
- Increase `embedding_dim` and `units`
- Use larger dataset
- Increase `epochs`
- Enable more regularization

## 🐛 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce `batch_size` or model size
2. **Slow Training**: Reduce dataset size or model complexity
3. **Poor Accuracy**: Increase training data or model capacity

### Dependencies
- TensorFlow 2.19.0+
- Python 3.8+
- 8GB+ RAM recommended

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- Built with TensorFlow/Keras
- Uses Bahdanau attention mechanism
- Optimized for CPU training on personal devices

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Review error messages
3. Ensure all dependencies are installed
4. Verify dataset format

---

**Happy Translating! 🌍➡️🇮🇳**

