# MSc in AI Capstone #5 - Generative AI Applications: HKDSE ICT Exam Question Generation

**Repository:** https://github.com/PHIacademy/generative_ai_applications.git

A Transformer-based conditional text generator, fine-tuned with LoRA in PyTorch, that writes novel multiple-choice questions in the style of the Hong Kong Diploma of Secondary Education (HKDSE) ICT Paper 1A examination. This project implements a base model and a LoRA-fine-tuned configuration that isolates the effect of fine-tuning, and a direct, evidence-based comparison of the two.

## Overview

- **Task type:** Conditional text generation (Transformer, decoder-only causal language model)
- **Dataset:** 560 real HKDSE ICT Paper 1A multiple-choice questions (2012–2025), published by the Hong Kong Examinations and Assessment Authority (HKEAA), distributed via a [Kaggle dataset](https://www.kaggle.com/datasets/stansir/dataset/data) prepared for this coursework
- **Base model:** TinyLlama-1.1B-Chat-v1.0, prompted directly with no fine-tuning
- **Fine-tuned model:** Identical base model and prompt format, with LoRA adapters (r=16, α=32) trained on the attention projections — the single variable isolated for comparison
- **Result:** The fine-tuned model produced a valid question format (stem + four options A–D) in **90%** of samples, against **0%** for the base model — see [Results](#results) below

## Project Structure

```
.
├── generative_model.ipynb                  # Main notebook: data prep → training → generation → evaluation
├── Generative_AI_Analysis_Report.pdf        # Written analysis report (APA 7 format)
├── requirements.txt                         # Reproducibility — exact package versions used
├── README.md
└── data/
    ├── final_dataset.csv                    # 560 HKDSE ICT Paper 1A questions
    ├── fields_explanation.md                # Column definitions for final_dataset.csv
    └── ict_curriculum.md                    # Official HKEAA ICT curriculum guide (classification reference)
```

## Dataset

560 HKDSE ICT Paper 1A multiple-choice questions, 2012–2025 (14 papers × 40 questions), published at https://www.kaggle.com/datasets/stansir/dataset/data. Each row holds the question stem, four answer options (A–D), the official answer key, image references where a figure accompanies the question, cohort percentage, and a `qn_type_id` classification against the official HKEAA *ICT Curriculum and Assessment Guide (Secondary 4–6)* (modules A–E). Field definitions are in `data/fields_explanation.md`.

Three questions with image-only answer options were excluded (options cannot be represented as text), leaving **557 usable questions**, split 445 train / 112 validation, stratified by curriculum module. The questions remain HKEAA copyright material; used here for non-commercial academic study only.

## Setup

**Requirements:** Python 3.10+, and a CUDA-capable GPU (recommended — training was run on 2× NVIDIA Tesla T4 on Kaggle; the notebook will also run on CPU, just considerably slower).

```
git clone https://github.com/PHIacademy/generative_ai_applications.git
cd generative_ai_applications
pip install -r requirements.txt
```

Or, from within the notebook itself:

```
%pip install -r requirements.txt
```

## Usage

Open `generative_model.ipynb` and run top to bottom. The notebook is organized into the following stages:

1. **Environment checks** — confirms required libraries import correctly and detects GPU/CPU
2. **Data loading and inspection** (Task 2) — loads `final_dataset.csv`, displays representative samples, inspects structure and token-length distribution, and documents preprocessing decisions (removing question numbers, replacing figure links, trimming PDF-extraction artefacts)
3. **Model and training** (Task 3) — loads TinyLlama-1.1B-Chat-v1.0, attaches LoRA adapters, trains with causal language-modelling loss, and plots loss curves
4. **Generation and evaluation** (Task 4) — generates samples from both the fine-tuned and base (adapter-off) models across 10 prompts spanning all five curriculum modules, computes format/consistency/copy-ratio metrics, and discusses strengths and failure cases with specific examples
5. **Summary** (Task 5) — key findings and caveats

Data paths are auto-detected for Kaggle (`/kaggle/input/...`) or a local `data/` folder; update `DATA_PATH` in the data-loading cell if your layout differs.

## Results

| Configuration | Valid format (stem + 4 options) | Module consistency* | Near-copy of a real question |
| -------------- | -------------------------------- | -------------------- | ------------------------------ |
| Base (adapter off) | 0 / 20 (**0%**) | 60% | 0 / 20 |
| Fine-tuned (LoRA) | 18 / 20 (**90%**) | 75% | 0 / 20 |

*Module consistency is measured by a TF-IDF nearest-centroid proxy, itself 82% accurate on real validation questions (chance = 20%).

Training loss fell from 2.09 to 0.88 over 5 epochs; validation loss stabilized at 0.957 (perplexity 2.6) by epoch 4, with no sign of over-fitting. The base model never produced a valid multiple-choice question in any of its 20 samples — it answered the conditioning header as a general-knowledge prompt instead. The fine-tuned model reliably reproduced DSE examination conventions (multi-statement questions, combination options, scenario stems), but close reading of the generated samples found only 1 of 10 usable without editing, due to overlapping distractors, missing single-correct answers, and occasional runaway generation. Full discussion, including example-by-example analysis, is in the notebook (Section 4.6) and the accompanying analysis report.

## Key Findings

- LoRA fine-tuning (0.41% of parameters trained) was necessary, not merely beneficial: the base model produced zero valid questions across 20 samples, while the fine-tuned model produced a correctly formatted question 90% of the time.
- Fine-tuning taught the model the *form* of an exam question far more reliably than its *substance* — generated distractors frequently overlap, lack a single defensible answer, or give the answer away, so every output needs expert review before use.
- Module conditioning (steering content toward the requested curriculum topic) is only weakly demonstrated: a 75%-vs-60% gap against a proxy metric that is itself only 82% accurate on real data.
- Careful data cleaning mattered: PDF-extraction artefacts (question numbers, image links, passage/footer text stuck after the last option) had to be stripped from training text, and the padding token had to be kept separate from the end-of-sequence token so the model could learn to stop.

## Limitations

- Small fine-tuning set (445 training examples across five curriculum modules)
- No figure-aware generation — 26% of real questions reference a diagram the text model cannot see or produce
- No answer-key generation or automated distractor-quality verification
- Evaluation metrics (format validity, TF-IDF module consistency, copy ratio) measure form, not educational correctness; qualitative review by a subject-matter expert remains necessary

See the analysis report for a full discussion of limitations, ethical considerations, and future improvements.

## Reproducibility

`requirements.txt` was generated via `pip freeze` after a full successful run of `generative_model.ipynb` on Kaggle, capturing the exact package versions used.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017.*
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022.*
- Zhang, P., Zeng, G., Wang, T., & Lu, W. (2024). TinyLlama: An open-source small language model. *arXiv:2401.02385.*
- Hong Kong Examinations and Assessment Authority. (2022). *ICT curriculum and assessment guide (Secondary 4–6).* Education Bureau, HKSARG.

Full citation list is available in `Generative_AI_Analysis_Report.pdf`.

## License

This project was completed as part of an MSc in Artificial Intelligence capstone assignment. The question dataset is sourced from official HKEAA examination papers and is used for non-commercial academic study only; see the analysis report for full sourcing details.
