# Kyrgyz News Summarization using mT5 and Extractive Methods

This repository provides an implementation for summarizing Kyrgyz news articles using both **abstractive** and **extractive** summarization techniques. The goal of the project is to build models capable of generating concise and coherent summaries of news articles in Kyrgyz, making it easier for readers to grasp key information quickly.

The project utilizes the **Google mT5** model for abstractive summarization, a pre-trained multilingual variant of the T5 (Text-to-Text Transfer Transformer) model, which has shown promising results in text generation tasks. For extractive summarization, traditional **NLP** techniques such as **TF-IDF**, **TextRank**, and **BERT-based** models are employed to select the most relevant sentences from the articles.

### **Project Components**
1. **Abstractive Summarization with mT5**:
   - Fine-tuning the `mT5` model on a dataset of Kyrgyz news articles to generate summaries.
   - The `mT5` model is trained to produce new, fluent sentences that convey the most important information from the original article.

2. **Extractive Summarization**:
   - Utilizing techniques like **TF-IDF**, **TextRank**, and **BERT-based models** for extracting key sentences from the articles.
   - The extractive approach aims to select the most informative and representative sentences directly from the input article.

3. **Evaluation**:
   - The models are evaluated using standard text summarization metrics like **ROUGE**, **BLEU**, and **BERTScore** to assess the quality of the generated summaries.

### **Dataset**
The dataset used in this project consists of Kyrgyz news articles, paired with human-generated summaries. The articles cover various topics, including politics, economy, culture, and sports, providing a broad context for testing the summarization models.

- `news.csv`: Contains the Kyrgyz news articles and their corresponding summaries (ground truth).
- `my_data.csv`: An additional dataset used for testing and validation.

### **Objectives**
- To develop a robust and efficient system for summarizing news articles in Kyrgyz, catering to both extractive and abstractive approaches.
- To explore the capabilities of the `mT5` model in the context of a lesser-represented language.
- To provide an open-source framework that can be extended and adapted to other languages or domains.

### **Techniques & Libraries Used**
- **Abstractive Summarization**: `mT5` (Pre-trained multilingual T5 model), **Hugging Face Transformers** library
- **Extractive Summarization**: **TF-IDF**, **TextRank**, **BERT-based models**
- **Evaluation Metrics**: **ROUGE**, **BLEU**, **BERTScore**
- **Libraries**: 
  - `transformers`, `torch`, `pandas`, `nltk`, `sklearn`, `matplotlib`, `seaborn`
  - **Hugging Face** and **TensorFlow** for model training and inference.

### **Installation Instructions**
To set up the project and run the models, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/YOUR_USERNAME/Kyrgyz_News_Summarization.git
   cd Kyrgyz_News_Summarization
