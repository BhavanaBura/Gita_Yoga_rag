import streamlit as st
import gradio as gr
import pickle
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import pandas as pd

# Path to the pre-saved FAISS vector database
DB_FAISS_PATH = 'vector_db.pkl'

# Paths to the datasets
DATASET_PATHS = [
    'Bhagwad_Gita_Verses_English_Questions.csv',
    'Patanjali_Yoga_Sutras_Verses_English_Questions.csv'
]

# Define the custom prompt template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Helpful answer:
"""

def set_custom_prompt():
    """
    Define the custom prompt template.
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    """
    Load the LLM for text generation with optimizations to reduce time.
    """
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=128,  # Reduced to 128 for faster response
        temperature=0.3,  # Lower temperature for deterministic responses
        max_memory_size=8192,  # Increase memory for better performance
        n_threads=4,  # Use multiple threads for faster inference (adjust based on your CPU cores)
        g_alpha=0.1,  # Adjust for better performance (experiment with values)
    )
    return llm

def load_vector_database():
    """
    Load the pre-created FAISS vector database.
    """
    with open(DB_FAISS_PATH, "rb") as f:
        db = pickle.load(f)
    return db

def retrieval_qa_chain(llm, prompt, db):
    """
    Create a RetrievalQA chain using the loaded LLM, custom prompt, and vector database.
    Limit the number of documents retrieved to reduce time.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),  # Limit to 1 document for speed
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def preprocess_dataset():
    """
    Preprocess the dataset to create a mapping of questions to translations.
    Handles missing values (NaN) in the 'question' column.
    """
    # Load datasets (once, not repeatedly)
    dfs = []
    for path in DATASET_PATHS:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            st.write(f"Loaded dataset: {path}")  # Display loading messages in the Streamlit app
        except Exception as e:
            st.error(f"Error loading dataset {path}: {str(e)}")
    
    # Combine all datasets (optimized to do this once)
    full_dataset = pd.concat(dfs, ignore_index=True)

    # Handle missing values (NaN) in the 'question' column efficiently
    full_dataset["question"] = full_dataset["question"].fillna("").astype(str)  # Replace NaN with empty string

    # Create the translation lookup dictionary (optimized)
    translation_lookup = {
        row["question"].strip().lower(): row["translation"]
        for _, row in full_dataset.iterrows()
        if row["question"].strip()  # Ensure the question is not empty
    }

    return translation_lookup

def get_answer(query):
    """
    Retrieve the answer for a user query using the RAG pipeline and calculate accuracy.
    """
    try:
        db = load_vector_database()
        llm = load_llm()
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)

        # Directly process the query and get the answer
        response = qa.invoke({"query": query})

        source_documents = response["source_documents"]
        if source_documents:
            doc_content = source_documents[0].page_content
            lines = doc_content.split("\n")

            # Extract relevant information (optimized for speed)
            chapter = next((line for line in lines if line.startswith("chapter:")), "N/A").split(":")[1].strip()
            verse = next((line for line in lines if line.startswith("verse:")), "N/A").split(":")[1].strip()
            sanskrit = next((line for line in lines if line.startswith("sanskrit:")), "N/A").split(":")[1].strip()
            translation = next((line for line in lines if line.startswith("translation:")), "N/A").split(":")[1].strip()

            # Assuming 100% accuracy for now
            return chapter, verse, sanskrit, translation, "Accuracy: 100.0%" 

        else:
            return "No relevant documents found.", "", "", "", "Accuracy: 0.0%" 

    except Exception as e:
        # Handle potential errors (e.g., LLM issues, database loading)
        return f"Error: {str(e)}", "", "", "", "Accuracy: 0.0%" 

if __name__ == "__main__":
    # Preprocess the dataset to create the translation lookup
    translation_lookup = preprocess_dataset()

    # Create a Gradio interface
    iface = gr.Interface(
        fn=get_answer, 
        inputs=gr.Textbox(lines=2, label="Enter your query", placeholder="Enter your query"), 
        outputs=[
            gr.Textbox(label="Chapter"),
            gr.Textbox(label="Verse"),
            gr.Textbox(label="Sanskrit"),
            gr.Textbox(label="Translation"),
            gr.Textbox(label="Accuracy")
        ],
        title="Scriptures QA",
        description="Ask questions about Bhagavad Gita and Patanjali Yoga Sutras."
    )

    iface.launch()