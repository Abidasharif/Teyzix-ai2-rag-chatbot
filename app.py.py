from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st


import os
# Paste your token inside the quotes below
os.environ["HF_TOKEN"] = "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"

# 1. Set webpage configuration metadata (MUST be the very first streamlit command called)
st.set_page_config(page_title="TEYZIX Knowledge Chatbot", layout="wide")

# 2. Initialize persistent chat history array inside session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR DESIGN COMPONENT ---
with st.sidebar:
    st.title("🗂️ Knowledge Source")
    st.write("System configured to look at local directory indices:")
    st.info("Source: TEYZIX Internship Program – Updated Instructions and Policies.pdf")

    # Visual database validation state display
    # (Using the same path specified in Step 3/4)
    if os.path.exists("/Data/vectorstore/db_faiss/index.faiss"):
        st.success("✅ FAISS Knowledge Base Index Found & Active!")
    else:
        st.error("❌ FAISS Vector Store Index Missing. Run Ingestion Script first.")

# --- MAIN CHAT WINDOW DESIGN COMPONENT ---
st.title("🤖 TEYZIX CORE - Internal Knowledge Base")
st.subheader("AI-2 Domain Track Retrieval Chatbot")

# Render historical messages from state log
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Prompt Interface Controller
if user_query := st.chat_input("Ask a question about TEYZIX policies or guidelines..."):
    # 1. Display user query instantly in chat panel
    with st.chat_message("user"):
        st.markdown(user_query)

    # Append user question to session state tracking log
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # 2. Process system reply inside an AI loader display spinner block
    with st.chat_message("assistant"):
        with st.spinner("Retrieving document insights and synthesizing response..."):
            try:
                # Load context parsing embedding models
                embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                # Load the local FAISS index
                vector_db = FAISS.load_local(
                    "vectorstore/db_faiss",
                    embedding_model,
                    allow_dangerous_deserialization=True
                )

                # Execute context search matching (Fetch top 3 matches)
                retrieved_docs = vector_db.similarity_search(user_query, k=3)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # Assemble context query instructions template
                assembled_prompt = (
                    f"Using only the following context:\n{context_text}\n\n"
                    f"Answer the question: {user_query}"
                )

                # Call local Ollama instance (Make sure Ollama is running in the background!)
                llm = Ollama(model="llama3", temperature=0.0)
                ai_response = llm.invoke(assembled_prompt)

                # Display output string onto the chat screen
                st.markdown(ai_response)

                # Append answer to session state tracking log
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

            except Exception as e:
                st.error(f"An execution handling failure error occurred: {str(e)}")