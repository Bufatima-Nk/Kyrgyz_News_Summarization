# Kyrgyz News Summarization Project

This project involves training and evaluating several Natural Language Processing (NLP) models for the task of abstractive summarization and extractive summarization, specifically for news articles in Kyrgyz. The models used include mT5, mBART, and extractive-based methods. The goal of this project is to build models capable of summarizing Kyrgyz news articles effectively.

## Files in the Repository

### 1. **news.csv**
   - **Description**: This is the dataset containing Kyrgyz news articles and their corresponding summaries. The dataset is used to train and evaluate the summarization models.
   - **Columns**:
     - `content`: The full text of the news article.
     - `summary`: The human-written summary of the article.

### 2. **my_data.csv**
   - **Description**: Another dataset for summarization purposes, likely custom data containing additional news content in Kyrgyz, used for training or evaluation.

### 3. **mt5_abstractive_summarization_KG.ipynb**
   - **Description**: Jupyter Notebook for training an abstractive summarization model using mT5, a pre-trained model designed for sequence-to-sequence tasks. This notebook loads the `news.csv` dataset, preprocesses the data, fine-tunes the mT5 model, and evaluates it using various metrics like BLEU, ROUGE, and BertScore.
   - **Key Steps**:
     - Dataset Preprocessing
     - Model Initialization (`google/mt5-small`)
     - Model Training using the Trainer API
     - Evaluation on the validation set
     - Generation of summaries for sample news articles

### 4. **mbart_abstractive_summarization_KG.ipynb**
   - **Description**: Similar to the mT5 notebook, this Jupyter notebook trains an abstractive summarization model using mBART, a sequence-to-sequence model designed for multilingual text generation tasks. The notebook follows the same process of loading, preprocessing data, training the model, and evaluating the results.
   - **Key Steps**:
     - Dataset Preprocessing
     - Model Initialization (`facebook/mbart-large-cc25`)
     - Model Training and Evaluation

### 5. **extractive_summarization_KG.ipynb**
   - **Description**: This notebook focuses on extractive summarization techniques, where the model selects the most important sentences or parts from the news article to form a summary. It leverages extractive techniques like TF-IDF or transformers such as BERT for sentence extraction.
   - **Key Steps**:
     - Data Preprocessing for extractive summarization
     - Implementation of extractive methods like TF-IDF, BERT-based embeddings
     - Generation and Evaluation of extractive summaries

### 6. **README.md**
   - **Description**: The file you're reading now, which provides an overview of the project, its files, and how to run and use them.

## Requirements

To run the notebooks and models, you will need to install the following Python libraries:

```bash
pip install transformers torch scikit-learn rouge-score nltk bert-score matplotlib seaborn pandas
