# Kriol → English Neural Machine Translation

A complete Neural Machine Translation (NMT) system for translating Kriol to English, developed for the CDU IT CODE FAIR AI Challenge. This project implements an end-to-end pipeline using Facebook's NLLB-200 model with comprehensive data preprocessing, optimized training, and evaluation using COMET metrics.

## 🎯 Project Summary

This project addresses the challenge of translating Kriol (an Australian creole language) to English using state-of-the-art neural machine translation techniques. The system achieves a **COMET score of 0.7616** on a 200-sample evaluation set, demonstrating high-quality translation performance.

### Key Features
- **End-to-end NMT pipeline** using NLLB-200-distilled-600M
- **Comprehensive data preprocessing** with mojibake repair and normalization
- **Kriol language token integration** (kri_Latn) with intelligent embedding initialization
- **Optimized training** with mixed precision, gradient accumulation, and TensorBoard logging
- **Continue training functionality** for model refinement without full retraining
- **COMET evaluation** with quality-tuned decoding parameters
- **Data augmentation** using Kriol-English dictionary
- **GPU optimization** specifically tuned for RTX 5060 Laptop (8GB VRAM)

## 🛠️ Technologies Used

### Core Libraries
- **PyTorch 2.8.0+** - Deep learning framework
- **Transformers** - Hugging Face library for NLLB model
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Data splitting and preprocessing

### Specialized Libraries
- **COMET** - Translation quality evaluation
- **Sacremoses** - Text normalization
- **ftfy** - Text encoding repair
- **tqdm** - Progress bars
- **TensorBoard** - Training visualization

### Hardware Optimization
- **CUDA/GPU acceleration** with TF32 support
- **Mixed Precision Training** (BF16/FP16)
- **Gradient Accumulation** for memory efficiency
- **cuDNN benchmarking** for optimal kernel selection


## 🚀 Step-by-Step Implementation

### Step 1: Environment Setup
```python
# Enable GPU optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### Step 2: Data Loading and Validation
- Load parallel Kriol-English corpus from Excel/CSV
- Validate data integrity (nulls, duplicates, control characters)
- Generate comprehensive statistics and language detection

### Step 3: Text Preprocessing
- **Mojibake repair**: Fix encoding issues and CP1252 artifacts
- **Unicode normalization**: NFC normalization and control character removal
- **Text cleaning**: Sentence punctuation and casing normalization
- **Length filtering**: Remove sentences exceeding token limits

### Step 4: Data Splitting and Augmentation
- Split data into training (80%) and validation (20%) sets
- **Dictionary augmentation**: Add Kriol-English word pairs and simple sentence patterns
- Increase training data from ~18K to ~23K pairs

### Step 5: Tokenization Analysis
- Analyze Kriol tokenization using NLLB tokenizer
- Verify acceptable unknown token rate (<10%)
- Plan vocabulary extension if needed

### Step 6: Language Token Integration
- Add new Kriol language token (`kri_Latn`) to tokenizer
- Initialize new token embeddings using English (`eng_Latn`) embeddings
- Resize model embeddings to accommodate new vocabulary

### Step 7: Model Setup
- Load NLLB-200-distilled-600M model
- Configure model for Kriol-English translation
- Set up GPU device and mixed precision training

### Step 8: Training Configuration
- **Optimizer**: Adafactor with constant learning rate schedule
- **Batch size**: 16 (optimized for 8GB VRAM)
- **Mixed precision**: BF16 when supported, FP16 with GradScaler
- **Gradient accumulation**: 2 steps to simulate larger batches

### Step 9: Main Training Loop (50,000 steps)
- **Manual training loop** with comprehensive error handling
- **TensorBoard logging** for loss and learning rate tracking
- **Checkpoint saving** every 1,000 steps
- **Memory management** with automatic cleanup
- **Progress tracking** with tqdm

### Step 10: Continue Training (5,000 steps)
- Resume training from final checkpoint
- Lower learning rate (5e-5) for fine-tuning
- Additional refinement without full retraining

### Step 11: Model Saving
- Save final model artifacts (weights, config, tokenizer)
- Export PyTorch state dictionary (.pth file)
- Save training loss history
- Generate fast tokenizer for inference

### Step 12: Model Testing
- Load final model and test on sample data
- Implement GPU-optimized translation function
- Quality-tuned decoding parameters:
  - `num_beams=12`, `length_penalty=1.20`
  - `repetition_penalty=1.05`, `no_repeat_ngram_size=3`
  - N-best reranking by sequence scores

### Step 13: COMET Evaluation
- **Length-aware sampling**: Select 200 longer Kriol sentences
- **Quality decoding**: 16 beams with optimized parameters
- **Reference formatting**: Ensure proper COMET input format
- **Batch evaluation**: Efficient GPU-based scoring

## 📊 Results

- **Training Loss**: Reduced from 8.10 to 0.56 over 50,000 steps
- **COMET Score**: **0.7616** on 200-sample evaluation
- **Training Time**: ~20 hours for main training, ~7 hours for continue training
- **Model Size**: ~600M parameters (NLLB-200-distilled)

## 🔧 Key Optimizations

### GPU Performance
- **TF32**: Enabled for Ampere+ GPUs (RTX 5060)
- **Mixed Precision**: BF16/FP16 with automatic scaling
- **cuDNN Benchmark**: Automatic kernel optimization
- **Non-blocking transfers**: Asynchronous CPU-GPU data movement

### Memory Management
- **Gradient accumulation**: Simulate larger batch sizes
- **Automatic cleanup**: Regular GPU memory clearing
- **Checkpoint management**: Keep only recent checkpoints
- **Efficient data loading**: Pre-tokenized datasets

### Training Stability
- **Error handling**: Robust RuntimeError recovery
- **Learning rate scheduling**: Constant with warmup
- **Gradient clipping**: Prevent exploding gradients
- **Early stopping**: Based on validation metrics

## 🚀 Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training
1. Place your Kriol-English data in `data/train_data.xlsx`
2. Run the notebook cells sequentially
3. Monitor training progress with TensorBoard:
   ```bash
   tensorboard --logdir model/tb
   ```

### Evaluation
1. Run the COMET evaluation cell
2. Check results in `data/test.csv`
3. View model performance metrics

### Inference
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load trained model
tokenizer = AutoTokenizer.from_pretrained("model/final")
model = AutoModelForSeq2SeqLM.from_pretrained("model/final")

# Translate
tokenizer.src_lang = "kri_Latn"
tokenizer.tgt_lang = "eng_Latn"
inputs = tokenizer("your kriol text", return_tensors="pt")
outputs = model.generate(**inputs)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| COMET Score | 0.7616 |
| Training Steps | 55,000 total |
| Final Loss | 0.56 |
| Training Data | 23,692 pairs |
| Model Parameters | ~600M |
| GPU Memory | 8GB VRAM |

## 🎯 Future Improvements

- **Larger model**: Upgrade to NLLB-200-1.3B or 3B parameters
- **More training data**: Expand corpus with additional Kriol sources
- **Advanced techniques**: Implement LoRA or other parameter-efficient fine-tuning
- **Domain adaptation**: Specialize for specific Kriol dialects
- **Real-time inference**: Optimize for production deployment

## 📚 References

- [NLLB Model Card](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [Transformers Seq2Seq Documentation](https://huggingface.co/docs/transformers/en/tasks/translation)
- [Fine-tuning NLLB Tutorial](https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865)
- [COMET Evaluation](https://github.com/Unbabel/COMET)

## 📄 License

This project is developed for the CDU IT CODE FAIR AI Challenge. Please refer to the competition guidelines for usage terms.

---

**Developed for CDU IT CODE FAIR AI Challenge**
