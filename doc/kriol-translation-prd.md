# PRD — Kriol→English NMT (Single model, single‑GPU)

## 1) Objective & Metric
- Objective: Build a Kriol→English translator.
- Primary metric: COMET on a fixed validation split.

## 2) Scope (what we will and won’t do)
- Model: NLLB (facebook/nllb-200-distilled-600M) fine-tuned for Kriol→English.
- Language code: Use `tpi_Latn` as tokenizer proxy for Kriol (NLLB lacks Kriol).
- Single-GPU training via Hugging Face Trainer; no multi-GPU or multi-env.
- Use English→Kriol-proxy (EN→tpi) to generate synthetic data for augmentation (not delivered as a product).

## 3) Data Pipeline
1. Load pairs from `data/train_data.xlsx` using columns `kriol` (SRC) and `english` (TGT).
2. Cleaning step writes `data/train_data_cleaned.csv`. If it exists, skip re-cleaning and load it directly.
3. Preprocess/normalize before split:
   - Quotes/whitespace normalization.
   - Kriol (source): lowercase; optional punctuation strip.
   - English (target): keep casing; normalize spacing; optional punctuation strip.
   - Length filter (≤128 tokens/side) and length-ratio filter (≤3.0 both ways).
   - Deduplicate exact pairs.
   - Optional: English LID check on targets (default: ON for baseline).
4. Split: 5-fold CV with fixed seed (default fold=0).

## 4) Training Stages
Stage A — Baseline fine-tune
- Tokenizer language codes: `SRC=tpi_Latn`, `TGT=eng_Latn`.
- Trainer config: AdamW (adamw_torch), LR=3e-5, warmup_steps=500, label_smoothing=0.1, gradient_accumulation to reach effective batch ≈16, fp16 if CUDA, predict_with_generate=True, group_by_length=True.
- Regularization: set available model dropouts (dropout≈0.2, attention_dropout≈0.1).
- Checkpointing: save `final/` under `model/`.
- Logging: TensorBoard enabled; save `CFG.json` for reproducibility.

Stage B — Back-translation augmentation
- Generate English→Tok Pisin (tpi) synthetic with NLLB (beam_size=8) in chunks.
- Save CSV to `data/synthetic/en_to_kriol_v2.csv` and a 100-row preview `_sample.csv`.
- Integrate into TRAIN ONLY with same normalization/filters and dedup; cap synthetic:real up to 1:1.
- Optional round-trip filter using the current model and chrF threshold.

Stage C — Data augmentation
- Apply token-level noise to Kriol (random drops/swaps) and light synonym replacement to English (WordNet), ~20% of train.

Stage D — Retrain on merged set
- Re-train with same hyperparameters, early stopping.

Optional — Custom tokenizer (SentencePiece)
- Train shared unigram SentencePiece tokenizer over combined Kriol+English corpus with vocab_size=10k.
- Save artifacts to `outputs/tokenizers/spm_kriol_en_v1`.
- Adoption criteria: Switch only if OOV >1.5% or A/B shows COMET improvement. Keep NLLB tokenizer otherwise.

## 5) Evaluation & Artifacts
- COMET on validation for `final/`; save:
  - `final/system_score.txt`, `final/comet_segments.csv`
- Inference helper: translate sample inputs with `final/`.

## 6) Acceptance Criteria
- End-to-end run completes without errors (baseline and augmented runs).
- Artifacts exist: `model/final/` with HF files + `model_state.pth`.
- COMET evaluation produces system score files and per-segment CSVs.
- Synthetic integration respects normalization, filters, dedup, and ≤1:1 cap.
- If SPM is adopted, A/B includes COMET delta and qualitative examples.

## 7) Roadmap (checklist)
- [x] Import data
- [x] Preprocess + split (train/val via CV)
- [x] Baseline train (save final, TensorBoard)
- [x] Generate EN→tpi synthetic (CSV + preview)
- [x] Integrate synthetic (cap ≤1:1, hygiene)
- [x] Add augmentation (noise + synonyms)
- [x] Retrain on merged set
- [x] Step 16 — COMET evaluation (final only)
- [ ] Push repo to GitHub master with README and environment info


— End —
