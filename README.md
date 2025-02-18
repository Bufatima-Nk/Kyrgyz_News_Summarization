# Kyrgyz News Summarization

## Overview
This project focuses on summarizing Kyrgyz news articles using both **extractive** and **abstractive** summarization techniques. It includes implementations of **mBART**, **MT5**, and **TF-IDF-based extractive methods**.

## Methods Used
- **Extractive Summarization**: Uses FastText embeddings, TF-IDF, cosine similarity, and position-based scoring to extract key sentences.
- **Abstractive Summarization**: Uses pre-trained transformer models **MT5** and **mBART** for generating summaries in the Kyrgyz language.

## Dataset
The project uses two datasets:
- `news.csv`: Contains raw Kyrgyz news articles.
- `my_data.csv`: A cleaned and formatted version of the news dataset for training and testing.

## Dependencies
To run the Jupyter notebooks, install the required dependencies:

```bash
pip install bert_score nltk rouge-score gradio networkx scikit-learn pandas numpy torch transformers
