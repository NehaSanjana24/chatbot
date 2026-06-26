import json
import os
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ==========================================
# STEP 1: Dependencies & NLTK Setup
# ==========================================
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ==========================================
# STEP 2: Load Dynamic JSON Dataset
# ==========================================
DATASET_PATH = "Ecommerce_FAQ_Chatbot_dataset.json"

@st.cache_data
def load_knowledge_base(file_path):
    """Loads the FAQ dataset from a JSON file and processes it into structures."""
    if not os.path.exists(file_path):
        # Fallback dictionary if the file is missing during initialization
        return {"Error": "Dataset file not found."}, ["Error"]
        
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    kb = {}
    for item in data.get("questions", []):
        q = item.get("question")
        a = item.get("answer")
        if q and a:
            kb[q] = a
            
    return kb, list(kb.keys())

# Load the data
faq_knowledge_base, faq_questions = load_knowledge_base(DATASET_PATH)

# ==========================================
# STEP 3: NLP Text Preprocessing Pipeline
# ==========================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in string.punctuation and token not in stop_words
    ]
    return " ".join(cleaned_tokens)

# Precompute vectors for your newly loaded dataset items to maximize response speed
if faq_questions and faq_questions[0] != "Error":
    preprocessed_faqs = [preprocess_text(q) for q in faq_questions]
else:
    preprocessed_faqs = []

# ==========================================
# STEP 4: Matching Logic via TF-IDF Vectors
# ==========================================
def get_chatbot_response(user_query, similarity_threshold=0.25):
    if not preprocessed_faqs or faq_questions[0] == "Error":
        return f"System configuration error: Please verify that '{DATASET_PATH}' exists in the workspace directory."

    cleaned_query = preprocess_text(user_query)
    
    # Check if user query has valid words after cleaning
    if not cleaned_query.strip():
        return "I'm sorry, could you please provide more details or context in your question?"
        
    vectorizer = TfidfVectorizer()
    all_documents = preprocessed_faqs + [cleaned_query]
    tfidf_matrix = vectorizer.fit_transform(all_documents)
    
    faq_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, faq_vectors).flatten()
    
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[best_match_idx]
    
    # Match evaluation against threshold barrier
    if highest_score >= similarity_threshold:
        matched_question = faq_questions[best_match_idx]
        return faq_knowledge_base[matched_question]
    else:
        return "I'm sorry, I couldn't find an answer to that specific question. Could you please contact customer support or rephrase it?"

# ==========================================
# STEP 5: Streamlit Web UI Layout
# ==========================================
st.set_page_config(page_title="E-commerce Support Bot", page_icon="🤖")

st.title("🤖 E-commerce FAQ Chatbot")
st.write(f"Currently querying a live database of **{len(faq_questions)}** automated support responses.")

# Keep message memory active across interface refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw structural elements for chat dialogue blocks
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input Bar Interaction
if user_query := st.chat_input("Ask a question (e.g., 'Can I cancel my order?', 'Do you ship to other countries?')"):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Calculate vector alignment and pick the top response
    reply = get_chatbot_response(user_query)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})