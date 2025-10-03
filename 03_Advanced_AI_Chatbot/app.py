# ==============================================================================
# PHASE 1: SETUP & IMPORTS
# ==============================================================================
import streamlit as st
import os
import pandas as pd
import faiss
import re
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google.cloud import dialogflow_v2 as dialogflow
import uuid

# --- CRITICAL: SET GOOGLE CREDENTIALS ---
credentials_path = "google_credentials.json"
if not os.path.exists(credentials_path):
    st.error(f"CRITICAL ERROR: The credentials file '{credentials_path}' was not found. Please add the file to your project folder.")
    st.stop()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# --- Page Configuration ---
st.set_page_config(page_title="Advanced AI Support Chatbot", page_icon="🤖", layout="centered")

# ==============================================================================
# PHASE 2: LOAD ASSETS & CONFIGURE APIS
# ==============================================================================
load_dotenv()

# --- Load API Keys & Config ---
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
DIALOGFLOW_REGION = os.getenv("DIALOGFLOW_REGION") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = "Table 1"

# --- Configure APIs ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    GEMINI_MODEL = genai.GenerativeModel('models/gemini-2.5-flash-lite', safety_settings=safety_settings)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# --- Cached functions to load heavy assets ---
@st.cache_resource
def load_encoder_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
@st.cache_resource
def load_faiss_index():
    return faiss.read_index('knowledge_base.index')
@st.cache_data
def load_knowledge_base():
    return pd.read_csv('knowledge_base_100k_csv_sample.csv')

try:
    ENCODER_MODEL = load_encoder_model()
    FAISS_INDEX = load_faiss_index()
    KNOWLEDGE_BASE_DF = load_knowledge_base()
except FileNotFoundError as e:
    st.error(f"A required file was not found: {e}.")
    st.stop()

# ==============================================================================
# PHASE 3: CORE AI & BUSINESS LOGIC FUNCTIONS
# ==============================================================================

def detect_intent(project_id, session_id, text, language_code='en', region=None):
    """Our "Smart Waiter" calling the "Snack Bar" (Dialogflow)."""
    # --- THIS IS THE FIX ---
    # We explicitly tell the client which "city" (region) to connect to.
    client_options = None
    if region and region != "global":
        client_options = {"api_endpoint": f"{region}-dialogflow.googleapis.com"}

    session_client = dialogflow.SessionsClient(client_options=client_options)
    session = session_client.session_path(project_id, session_id)
    text_input = dialogflow.TextInput(text=text, language_code=language_code)
    query_input = dialogflow.QueryInput(text=text_input)
    try:
        response = session_client.detect_intent(request={"session": session, "query_input": query_input})
        return response.query_result.intent.display_name, response.query_result.fulfillment_text
    except Exception as e:
        print(f"Error in Dialogflow call: {e}")
        return "Default Fallback Intent", "I'm sorry, my brain is having connection issues. Please try again."

# ... (The other functions remain exactly the same) ...
def find_relevant_context(user_question, top_k=5):
    cleaned_question = re.sub(r'@\w+', '', user_question).strip().lower()
    question_embedding = ENCODER_MODEL.encode([cleaned_question])
    distances, indices = FAISS_INDEX.search(question_embedding, top_k)
    return KNOWLEDGE_BASE_DF.iloc[indices[0]]

def generate_smart_answer(user_question, context_df):
    context_text = "\n".join([f"Q: {row['question']}\nA: {row['answer']}" for index, row in context_df.iterrows()])
    prompt = f"""You are an expert customer support agent. Answer the user's question based ONLY on the provided historical context. Be concise and helpful. If the context is not enough, say so.

    Context:\n---\n{context_text}\n---\n
    User's Question: "{user_question}"

    Answer:"""
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, there was an error with the AI model: {e}"

def create_support_ticket(question_text):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    data = {"records": [{"fields": {"User Question": question_text, "Status": "Todo"}}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        return response.status_code == 200
    except Exception:
        return False

# ==============================================================================
# PHASE 4: STREAMLIT UI & CHAT LOGIC
# ==============================================================================
st.title("🤖 Advanced AI Support Chatbot")
st.write("Ask me anything about customer support issues from our knowledge base!")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # We now pass the region to our detect_intent function
            intent_name, fulfillment_text = detect_intent(GOOGLE_PROJECT_ID, st.session_state.session_id, user_input, region=DIALOGFLOW_REGION)
            
            final_response = ""
            if intent_name == 'RequestHuman':
                ticket_created = create_support_ticket(user_input)
                final_response = "I've created a support ticket for you. An agent will be in touch shortly." if ticket_created else "Sorry, I couldn't create a ticket right now."
            elif intent_name == 'Default Fallback Intent':
                context = find_relevant_context(user_input)
                final_response = generate_smart_answer(user_input, context)
            else:
                final_response = fulfillment_text
        st.markdown(final_response)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
