#Gita_Yoga_rag
# Retrieval-Augmented Generation (RAG) Pipeline for Bhagavad Gita and Patanjali Yoga Sutras

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, designed to answer user queries about the Bhagavad Gita and Patanjali Yoga Sutras. It leverages a FAISS vector database to efficiently retrieve relevant verses and uses a language model (LLM) to generate meaningful responses based on these translations.

## Project Overview

The project is divided into two main parts:

1. Ingestion (Dataset Preparation): This part loads the CSV datasets containing the verses and questions from the Bhagavad Gita and Patanjali Yoga Sutras. It splits the text into smaller chunks for better processing and creates a FAISS vector database.
   
2. Retrieval and Answer Generation: In this part, the pre-built vector database is used to retrieve relevant verses based on a user's query. A language model generates a response using the retrieved verses as context.

The user interface is built using Gradio, which allows users to easily interact with the system, input their queries, and receive answers.

## Requirements

- Python 3.8+
- Install the required Python packages:
  ```bash
  pip install langchain streamlit gradio pandas langchain-huggingface faiss-cpu
