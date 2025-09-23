# Product Requirements Document
## Kriol→English Neural Machine Translation System

### 1. Executive Summary

This PRD outlines the development of a neural machine translation system for translating Kriol to English, leveraging MarianMT as the baseline model with progressive enhancements including synthetic data generation, preprocessing optimizations, and advanced techniques.

### 2. Project Overview

**Objective**: Build a robust Kriol→English translation model for a low-resource language challenge  
**Primary Metric**: COMET score  
**Core Approach**: MarianMT fine-tuning with data augmentation  
**Key Constraint**: Limited parallel training data  

### 3. Technical Implementation Roadmap

#### Phase 1: Minimal Viable Baseline

**3.1.1 Minimal MarianMT Setup**
- Use a pre-trained MarianMT checkpoint from Hugging Face for Kriol↔English or closest available variant.
- Phase 1 keeps MarianMT's built-in tokenizer and vocabulary (no custom swap yet).
- Disable advanced features (no back-translation, no augmentation, no ensembling).
- Decoding: greedy (beam size = 1) for speed; max length 128.
- Save a .pth checkpoint and ensure the notebook/script can load and run inference end-to-end.

**3.1.2 Tokenizer Training Plan**
- Note: Phase 1 uses MarianMT's default tokenizer. This plan is prepared for later phases if we decide to adopt a shared custom tokenizer.
- Approach: Train a shared SentencePiece tokenizer (unigram) over Kriol and English sides combined.
- Corpus: All available training sentences from both languages; include allowed public monolingual data later if used.
- Preprocessing:
  - Lowercase
  - Normalize whitespace and punctuation; retain dialectal spellings
  - Remove duplicates and empty lines
- Hyperparameters:
  - Vocabulary size: 8000 (expand to 12000 if OOV rate > 1.5%)
  - Character coverage: 0.9995
  - Model type: unigram (fallback: BPE)
  - Special/control tokens: <pad>, <s>, </s>, <unk>
- Training flags (indicative):
  - --model_type=unigram --vocab_size=8000 --character_coverage=0.9995
  - --shuffle_input_sentence=true --max_sentence_length=2048 --num_threads=[CPU cores]
- Sharing policy: Use one shared tokenizer for both languages in Phase 1–2; retrain only if corpus grows by >30% or OOV >1.5%.
- Artifacts & versioning: spm.model, spm.vocab saved as spm_kriol_en_v1; log training config; freeze for reproducibility.
- OOV handling: Allow copy-through of unknown spans; add glossary later in Phase 4.

**3.1.3 Data Preprocessing & Normalization**
- Language ID filter to remove non-Kriol/English lines
- Length and ratio filters (max length 128; src/tgt length ratio ≤ 3.0)
- Deduplication across train/val to avoid leakage

**3.1.4 Initial Training**
- Optimizer: AdamW; Learning rate: 5e-5; Warmup steps: 500
- Batch size: 16–32 (use gradient accumulation if needed)
- Label smoothing: 0.1; Mixed precision if available
- Train 1 epoch to validate the end-to-end pipeline; focus on reproducible run and COMET evaluation

**3.1.5 Evaluation**
- Compute COMET on validation set; store baseline score and decoded samples

#### Phase 2: Baseline Enhancement (MarianMT Fine-Tuning)

**3.2.1 MarianMT Setup**
- Use pre-trained MarianMT model as base
- Configure for Kriol→English fine-tuning
- Set up evaluation pipeline with COMET metric

**3.2.2 Data Preprocessing**
- Keep MarianMT's built-in tokenizer (no custom swap in this phase)
- Text normalization:
  - Convert to lowercase
  - Unify spelling variations
  - Handle contractions and dialectal variance
  - Remove noise while preserving dialectal nuances
- Deduplication and split hygiene; language ID and length/ratio filters

**3.2.3 Initial Training**
- Baseline hyperparameters:
  - Learning rate: 2e-5 to 5e-5
  - Batch size: 16–32 (adjust based on GPU memory)
  - Dropout: 0.1
  - Warmup steps: 500–1000

**3.2.4 Decoding & Checkpointing**
- Beam search: beam size 4–6; length penalty 1.0; early stopping enabled
- Save checkpoints every 1000 steps; keep top-3 by validation COMET
- Select best checkpoint by COMET on a fixed validation set

#### Phase 3: Data Augmentation & Synthetic Data Generation

**3.3.1 Back-Translation Pipeline**
- Train initial English→Kriol model using MarianMT
- Generate synthetic Kriol→English pairs:
  1. Translate English corpus to Kriol
  2. Add generated pairs back to training set
  3. Weight synthetic data appropriately (0.5-0.7x real data weight)

**3.3.2 Orthographic Augmentation**
- Create spelling variations:
  - Common misspellings
  - Phonetic alternatives
  - Dialectal variants
- Generate 2-3 variations per sentence

**3.3.3 Synonym Substitution**
- Use WordNet or similar for English side
- Replace 10-20% of words with synonyms
- Maintain sentence structure and meaning

**3.3.4 Data Quality Control**
- Filter synthetic data using confidence scores
- Remove pairs with <0.3 confidence
- Manual spot-check of generated samples

**3.3.5 Tokenizer Adoption (Optional)**
- If synthetic data increases vocabulary pressure (OOV > 1.5% or corpus +30%), adopt the shared SentencePiece tokenizer from 3.1.2 and compare performance against MarianMT's default tokenizer.

#### Phase 4: Medium Improvements

**3.4.1 Hyperparameter Optimization**
- Grid search on:
  - Learning rate: [1e-4, 3e-4, 5e-4, 7e-4]
  - Batch size: [16, 32, 64]
  - Dropout: [0.1, 0.2, 0.3]
  - Warmup strategies
- Use validation set for early stopping

**3.4.2 Rare Word Handling**
- Implement copy mechanism for OOV words
- Build glossary of common untranslatable terms
- Add character-level fallback for unknown tokens

**3.4.3 Training Strategy Refinement**
- Curriculum learning: start with simpler sentences
- Mixed precision training for efficiency
- Checkpoint management (save every 1000 steps)

#### Phase 5: Advanced Enhancements [Optional]

**3.5.1 Transfer Learning**
- Incorporate related language data:
  - Haitian Creole→English
  - Tok Pisin→English
  - Jamaican Patois→English
- Multi-stage fine-tuning approach

**3.5.2 Model Ensembling**
- Train multiple model variants:
  - MarianMT baseline
  - mBART fine-tuned version
  - Different checkpoint epochs
- Combine via:
  - Weighted averaging
  - Majority voting
  - Reranking with COMET

**3.5.3 Multi-Task Learning**
- Add auxiliary tasks:
  - Masked language modeling on Kriol
  - Language identification
  - Part-of-speech tagging

### 4. Evaluation Strategy

**4.1 Metrics**
- Primary: COMET score
- Secondary: BLEU, chrF++
- Human evaluation samples (if feasible)

**4.2 Validation Approach**
- Hold-out validation set (10% of data)
- K-fold cross-validation for small datasets
- Compare multiple checkpoints (early vs late epochs)

### 5. Risk Mitigation

| Risk | Mitigation Strategy |
|------|-------------------|
| Synthetic data noise | Quality filtering, confidence thresholds, human validation |
| Overfitting | Dropout, early stopping, data augmentation |
| Domain mismatch | Collect diverse text types, balance formal/informal |
| Compute limitations | Start with smaller models, use cloud GPUs selectively |

### 6. Implementation Timeline

| Week | Focus | Deliverable |
|------|-------|------------|
| 2-3 | Data Augmentation | 2x expanded dataset with synthetic data |
| 3-4 | Optimization | Tuned model with 10-15% improvement |
| 4-5 | Advanced Features | Final submission-ready model |
| 5 | Final Testing | Documentation and submission |

### 7. Resource Requirements

**Compute**
- Minimum: 1x GPU with 8GB VRAM
- Recommended: 1x GPU with 16GB+ VRAM
- Training time: ~24-48 hours total

**Data Storage**
- Raw data: ~500MB
- Augmented data: ~2GB
- Model checkpoints: ~5GB

### 8. Success Criteria

- **Minimum**: COMET score improvement of >5% over baseline
- **Target**: COMET score improvement of >15% over baseline
- **Stretch**: Top 25% in challenge leaderboard

### 9. Monitoring & Logging

- TensorBoard for training metrics
- WandB for experiment tracking
- Version control for data and code
- Detailed experiment logs with hyperparameters

### 10. Post-Implementation

- Model deployment considerations
- Inference optimization
- Documentation of best practices
- Knowledge transfer to team

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]  
**Status**: Ready for Implementation