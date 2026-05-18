import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paste your token inside the quotes below
os.environ["HF_TOKEN"] = "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"

def create_local_vector_database():
    # File targets
    # Absolute path directly to your downloads folder structure
    pdf_source = "C:\\Users\\USER\\Downloads\\Teyzix Internship Files\\Task 2\\Data\\TEYZIX Internship Program – Updated Instructions and Policies.pdf"
    vector_destination = "vectorstore\\db_faiss"

    # Visual database validation state display
    # (Using the same path specified in Step 3/4)
    if os.path.exists("C:\\Users\\USER\\Downloads\\Teyzix Internship Files\\Task 2\\Data\\vectorstore\\db_faiss\\index.faiss"):
        print("✅ FAISS Knowledge Base Index Found & Active!")
    else:
        print("❌ FAISS Vector Store Index Missing. Run Ingestion Script first.")


    # --- STEP 2 Execution ---


    print("⏳ Starting document extraction pipeline...")
    loader = PyPDFLoader(pdf_source)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    print(f"📝 Preprocessed text into {len(chunks)} individual chunks.")

    # --- STEP 3 Execution ---

    print("🤖 Loading 'all-MiniLM-L6-v2' framework embeddings...")
    embedding_tool = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("🧠 Transforming text data into vector spaces via FAISS...")
    db_index = FAISS.from_documents(chunks, embedding_tool)

    print("💾 Committing vector index files locally to disk...")
    db_index.save_local(vector_destination)
    print("🎉 Success! Your local knowledge database is fully constructed and stored safely.")


if __name__ == "__main__":
    create_local_vector_database()

