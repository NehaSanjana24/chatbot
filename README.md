#E-commerce FAQ Chatbot

A Python-based intelligent FAQ Chatbot designed to handle e-commerce customer support questions. This application uses Natural Language Processing (NLP) to preprocess text, calculates similarity scores using TF-IDF vectorization and Cosine Similarity, and provides an interactive web interface built with Streamlit.

## 🚀 Live Demo
You can interact with the live deployed application here:
👉 [https://chatbot-nkyad2qctvaumxcwyszkea.streamlit.app/](https://chatbot-nkyad2qctvaumxcwyszkea.streamlit.app/)

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Frontend UI:** Streamlit
* **Natural Language Processing:** NLTK (Tokenization, Lemmatization, Stopwords removal)
* **Machine Learning Vectorization:** Scikit-learn (TfidfVectorizer, Cosine Similarity)
* **Dataset Format:** JSON

---

## 📋 Features & Workflow
1. **Dynamic Dataset Loading:** Parses customer service queries and matching answers straight from a structured `Ecommerce_FAQ_Chatbot_dataset.json` file.
2. **Text Preprocessing Pipeline:** 
   * Converts user text to lowercase.
   * Tokenizes phrases into separate words.
   * Removes punctuation marks and common English stop words.
   * Lemmatizes words to extract their base/root form.
3. **Intent Matching Engine:** Converts text into numerical vectors using **TF-IDF** and computes mathematical alignment using **Cosine Similarity**.
4. **Fallback Handling:** Implements a strict confidence threshold barrier (0.25) to cleanly refuse gibberish or unrelated inquiries.

---

## 📁 Repository Structure
```text
├── chatbot.py                          # Main application and UI script
├── Ecommerce_FAQ_Chatbot_dataset.json  # Structured Q&A dataset
├── requirements.txt                    # App dependencies for cloud deployment
└── README.md                           # Project documentation
