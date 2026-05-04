# Kyrgyz News Summarization — NLP for a Low-Resource Language

> Comparative study of abstractive and extractive summarization techniques for Kyrgyz-language news articles. Fine-tuned **mT5** and **mBART-50** transformer models, benchmarked against two extractive baselines, and evaluated across ROUGE, BLEU, METEOR, and BERTScore.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Language](https://img.shields.io/badge/Language-Kyrgyz%20(ky__KG)-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Why This Project Matters

Kyrgyz is a Turkic language spoken by ~5 million people, written in Cyrillic script, and highly agglutinative — making NLP significantly harder than for high-resource languages. Publicly available labeled summarization datasets for Kyrgyz are extremely scarce, and most existing multilingual models have minimal Kyrgyz pretraining coverage.

This project applies state-of-the-art multilingual transformer models to Kyrgyz news summarization, compares them against strong extractive baselines, and surfaces concrete insights about what works — and why — in low-resource NLP settings.

---

## Results

| Method | Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR | BERTScore |
|--------|-------|:-------:|:-------:|:-------:|:----:|:------:|:---------:|
| Extractive (FastText + TF-IDF scoring) | Custom | 0.458 | 0.200 | 0.441 | 0.089 | 0.446 | 0.741 |
| Extractive (TextRank + TF-IDF) | Custom | 0.366 | 0.115 | 0.312 | 0.099 | 0.361 | 0.742 |
| Abstractive | mT5-small | 0.193 | 0.026 | 0.191 | 0.081 | — | **0.919** |
| Abstractive | mBART-50 (6 epochs) | **0.303** | **0.081** | **0.295** | — | — | — |

### Key Findings

**1. Extractive methods outperform abstractive fine-tuning at this data scale.**
With limited labeled data, extractive approaches (which select existing sentences) consistently beat fine-tuned seq2seq models. This is expected — abstractive models require significantly more training examples to generate fluent, accurate summaries.

**2. mBART-50 substantially outperforms mT5-small (ROUGE-1: 30.3 vs 19.3).**
mBART-50-many-to-many was explicitly pretrained on Kyrgyz (`ky_KG`), giving it native morphological and lexical knowledge. mT5-small's Kyrgyz coverage in pretraining is minimal. Architecture alignment with the target language matters more than model size at low data regimes.

**3. mT5's high BERTScore (0.919) despite low ROUGE reveals a known low-resource NLP phenomenon.**
The model generates semantically coherent content that is not a surface-level match to the reference — plausible but paraphrased summaries. ROUGE penalizes this heavily; BERTScore captures semantic similarity and scores it highly. In practice, human evaluation would be needed to resolve which metric better reflects real quality.

**4. FastText + TF-IDF hybrid scoring (Extractive 1) beats pure TextRank (Extractive 2).**
Combining semantic sentence embeddings (FastText), positional bias, and keyword relevance (TF-IDF) outperforms graph-based TextRank across all ROUGE metrics, suggesting that positional heuristics are informative for Kyrgyz news structure.

---

## Approach

### Abstractive Summarization
Fine-tuned two pretrained multilingual seq2seq models on a Kyrgyz news dataset:

- **mT5-small** (`google/mt5-small`) — Text-to-Text Transfer Transformer, multilingual variant
  - Custom `KyrgyzNewsDataset` PyTorch class with proper padding and label masking
  - HuggingFace `Trainer` API, 10 epochs, lr=3e-5, batch size=4
  - Beam search decoding (num_beams=4)

- **mBART-50** (`facebook/mbart-large-50-many-to-many-mmt`) — Multilingual BART with native Kyrgyz support
  - `Seq2SeqTrainer` with `predict_with_generate=True`
  - fp16 mixed-precision training, 6 epochs
  - Source and target language explicitly set to `ky_KG`

### Extractive Summarization
Two custom extractive pipelines:

- **Method 1 — FastText Hybrid:** FastText sentence embeddings + cosine similarity to document centroid + TF-IDF keyword scoring + positional bias → weighted sentence ranking
- **Method 2 — TextRank:** TF-IDF vectorization → cosine similarity matrix → PageRank via NetworkX → top-k sentence extraction

### Evaluation
All models evaluated on the same held-out validation split using:
- **ROUGE-1/2/L** (surface n-gram overlap)
- **BLEU** (n-gram precision with brevity penalty)
- **METEOR** (synonym-aware recall)
- **BERTScore** (`bert-base-multilingual-cased`, lang=`ky`)

---

## Dataset

- **Source:** Kyrgyz news articles scraped from public news sources
- **Files:** `news.csv` (primary), `my_data.csv` (additional)
- **Columns:** `content` (full article text), `summary` (human-written summary)
- **Language:** Kyrgyz (ISO 639-1: `ky`, Cyrillic script)
- **Split:** 90% train / 10% validation (`random_state=42`)

---

## Project Structure

```
Kyrgyz_News_Summarization/
│
├── mt5_abstractive_summarization_KG.ipynb       # mT5 fine-tuning + evaluation
├── mbart_abstractive_summarization_KG.ipynb     # mBART-50 fine-tuning + evaluation
├── extractive_summarization_KG.ipynb            # Two extractive pipelines + Gradio demo
│
├── news.csv                                     # Primary dataset
├── my_data.csv                                  # Additional training data
│
├── requirements.txt                             # All dependencies
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Bufatima-Nk/Kyrgyz_News_Summarization
cd Kyrgyz_News_Summarization
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```bash
jupyter notebook
```

Open and run cells in any notebook:
- `extractive_summarization_KG.ipynb` — no GPU required, runs fast
- `mt5_abstractive_summarization_KG.ipynb` — GPU recommended (Google Colab works)
- `mbart_abstractive_summarization_KG.ipynb` — GPU required (mBART-50 is large)

### 4. Try the Gradio demo (extractive)
The last cell of `extractive_summarization_KG.ipynb` launches an interactive demo:
```python
interface.launch()  # Opens local UI to summarize any Kyrgyz text
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, HuggingFace Transformers, Datasets |
| Models | `google/mt5-small`, `facebook/mbart-large-50-many-to-many-mmt` |
| Extractive NLP | Gensim FastText, scikit-learn TF-IDF, NetworkX PageRank |
| Evaluation | rouge-score, NLTK (BLEU, METEOR), bert-score |
| Demo | Gradio |
| Data | pandas, NumPy |

---

## Limitations & Future Work

- **Dataset size** is the primary bottleneck. Abstractive models would likely improve significantly with 5–10× more labeled pairs.
- **mT5 results** could improve with longer training, larger model variant (mT5-base), or prefix tuning instead of full fine-tuning.
- **Human evaluation** is needed to validate BERTScore's optimistic assessment of mT5 outputs — automated metrics alone are insufficient for low-resource languages.
- **Deployment:** Both Gradio demos can be deployed to [HuggingFace Spaces](https://huggingface.co/spaces) for permanent public access.

---

## Author

**Bufatima N.K.**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bufatima--n--k-blue?logo=linkedin)](https://linkedin.com/in/bufatima-n-k)
[![GitHub](https://img.shields.io/badge/GitHub-Bufatima--Nk-black?logo=github)](https://github.com/Bufatima-Nk)
