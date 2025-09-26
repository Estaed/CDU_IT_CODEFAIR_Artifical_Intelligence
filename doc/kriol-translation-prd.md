# PRD — Kriol→English NMT (Single model, single‑GPU)

## 1) Objective & Metric
- Objective: Build a Kriol→English translator.
- Primary metric: COMET on a fixed validation split.

## 2) Scope (what we will and won’t do)
- Model: NLLB (facebook/nllb-200-distilled-600M) fine-tuned for Kriol→English.
- Single-GPU training via Hugging Face Trainer; no multi-GPU or multi-env.
- Use English→Kriol only offline to generate synthetic data for augmentation (not delivered as a product).

## 3) Data Pipeline
1. Load pairs from `data/train_data.xlsx` using columns `kriol` (SRC) and `english` (TGT).
2. Cleaning step writes `data/train_data_cleaned.csv`. If it exists, skip re-cleaning and load it directly.
3. Preprocess/normalize before split:
   - Quotes/whitespace normalization.
   - Kriol (source): strip punctuation + lowercase.
   - English (target): keep casing; normalize spacing only.
   - Length filter (≤128 tokens/side) and length-ratio filter (≤3.0 both ways).
   - Deduplicate exact pairs.
   - Optional: English LID check on targets (default: ON for our baseline; can be toggled).
4. Split: 90% train / 10% validation with fixed seed.

## 4) Training Stages
Stage A — Baseline fine-tune
- Trainer config: AdamW (adamw_torch), LR=5e-5, warmup_steps=500, label_smoothing=0.1, gradient_accumulation to reach effective batch ≈16, fp16 if CUDA, predict_with_generate=True.
- Checkpointing: save `final/` under `model/`.
- Logging: TensorBoard enabled; save `CFG.json` for reproducibility.
- Known HF quirk: avoid passing both `decoder_input_ids` and `decoder_inputs_embeds` during training. Our trainer sanitizes inputs to prevent this.

Stage B — Back-translation augmentation
- Generate English→Kriol synthetic with beam_size=4 (fast toggle: greedy) in chunks.
- Save CSV to `data/synthetic/en_to_kriol_v1.csv` and a 100-row preview `_sample.csv`.
- Integrate into TRAIN ONLY with same normalization/filters and dedup; cap synthetic:real ≈ 1:1.
- Retrain once on the merged train set; keep the same validation split.

## 5) Evaluation & Artifacts
- COMET on validation for `final/`; save:
  - `final/system_score.txt`, `final/comet_segments.csv`
- Inference helper: translate sample inputs with `final/`.

## 6) Optional (only if needed later)
- Custom SentencePiece tokenizer (shared unigram) if OOV pressure (>1.5%) or vocabulary coverage concerns arise. Otherwise, keep Marian tokenizer. If adopted: A/B compare COMET vs Marian tokenizer, then lock choice.

## 7) Acceptance Criteria
- End-to-end run completes without errors (baseline and augmented runs).
- Artifacts exist: `model/final/` with HF files + `model_state.pth`.
- COMET evaluation produces system score files and per-segment CSVs.
- Synthetic integration respects normalization, filters, dedup, and 1:1 cap.

## 8) Trackable Roadmap (checklist)
- [x] Import data
- [x] Preprocess + split (train/val)
- [x] Baseline train (save final, TensorBoard)
- [x] COMET baseline (save scores/segments)
- [x] Generate EN→Kriol synthetic (CSV + preview)
- [ ] Integrate synthetic (cap 1:1, hygiene)
- [ ] Retrain on merged set
- [ ] COMET after augmentation (save scores/segments)
- [x] Deliver inference helper + artifacts
- [ ] Push repo to GitHub master with README and environment info


— End —
