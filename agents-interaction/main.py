from sentence_transformers import SentenceTransformer
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.llms import LlamaCpp
import numpy as np
import time
import sys
import os
import torch
import faiss
import requests
import random
from llama_cpp import Llama

model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

torch.cuda.empty_cache()

# # Change the model name and store it in a variable
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
model_basename = "llama-2-13b-chat.Q5_K_S.gguf"  # the model is in bin format

# Change the model name and store it in a variable
# model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGUF"
# model_basename = "llama-2-7b-chat.Q2_K.gguf"  # the model is in bin format

# Change the FAISS index path to use the local file
DB_FAISS_PATH = "/usr/src/app/vectorstore/db_faiss/faiss.index"

# Download the model using HF-Hub
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

lcpp_llm = None
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,  # CPU cores
    n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=43,  # Change this value based on your model and your GPU VRAM pool.
    n_ctx=4096,  # Context window
)
# lcpp_llm2 = Llama(
#     model_path=model_path,
#     n_threads=2,  # CPU cores
#     n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#     n_gpu_layers=43,  # Change this value based on your model and your GPU VRAM pool.
#     n_ctx=4096,  # Context window
# )

# Load the dataset into a pandas DataFrame
loader = CSVLoader(file_path="/usr/src/app/data/mc-dataset.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()

# Change the text splitter import to use the langchain version
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=20
)

# Split the dataset into text chunks
text_chunks = text_splitter.split_documents(data)
text_strings = [doc.page_content for doc in text_chunks]

# Initialize the model from the sentence-transformers library
model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')


def get_embeddings(texts):
    try:
        # texts should be a list of strings
        embeddings = model.encode(texts, show_progress_bar=True)
        print("Encoding completed successfully.")
        return embeddings
    except Exception as e:
        print(f"An error occurred during encoding: {e}")
        return None


# Change the embedding import to use the langchain version
embeddings = get_embeddings(text_strings)


# Extract the embeddings from the FAISS index
# docsearch = FAISS.from_documents(text_chunks, embeddings)

# docsearch.save_local(DB_FAISS_PATH)
dimension = embeddings.shape[1]  # Get the dimensionality of your embeddings
index = faiss.IndexFlatL2(dimension)  # Create a flat (brute-force) search index

# FAISS requires the data type to be float32
if embeddings.dtype != np.float32:
    embeddings = embeddings.astype(np.float32)

index.add(embeddings)  # Add your embeddings to the index

# To save the index to disk
faiss.write_index(index, DB_FAISS_PATH)


def load_faiss_index(index_path):
    return faiss.read_index(index_path)


def fetch_question_from_api(question_id):
    """Fetch a question from the API."""
    api_url = f'http://3.139.84.244:8000/questions/{question_id}'
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching question: HTTP {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Exception when fetching question: {e}")
        return None


def generate_exam(num_questions):
    """Generate an exam with a specified number of questions."""
    exam = []
    question_ids = random.sample(range(1, 490), num_questions)  # Randomly select question IDs

    for q_id in question_ids:
        question = fetch_question_from_api(q_id)
        if question:
            exam.append(question)
        else:
            print(f"Failed to fetch question with ID: {q_id}")

    return exam


def generate_response(prompt):
    """Generate a response using the specified agent."""
    response = lcpp_llm(
        prompt=prompt,
        max_tokens=1000,
        temperature=0.5,
        top_p=0.95,
        repeat_penalty=1.2,
        top_k=50,
        stop=['USER:', 'ASSISTANT:'],
        echo=True
    )
    return response["choices"][0]["text"]


def agent_interaction(question, file_path, model, index=None):
    """
    Simulate the interaction based on a question object and store the chat history in the specified file.
    If an index is provided, perform a similarity search to assist with answering the question.
    """
    question_text = question['Question']
    choices_text = f"(A) {question['Choice_A']} (B) {question['Choice_B']} (C) {question['Choice_C']} (D) {question['Choice_D']} (E) {question['Choice_E']}"

    prompt_for_agent = f"Question: {question_text} Choices: {choices_text}\nAnswer:"

    # Perform similarity search if an index is provided
    if index is not None:
        question_embedding = model.encode([question_text], convert_to_tensor=True).numpy()
        D, I = index.search(question_embedding, k=1)  # Adjust k based on how many similar items you want to consider

        # Incorporate information from similar items into the prompt
        similar_item_info = " Based on similar documents, consider focusing on aspects related to the question."
        prompt_for_agent += similar_item_info

    response_from_agent = generate_response(prompt_for_agent)

    # Save the interaction to the specified chat history file
    save_chat_history(file_path, f"Question: {question_text}")
    save_chat_history(file_path, f"Agent's Response: {response_from_agent}")
    save_chat_history(file_path, f"Correct Answer: {question['Correct_Answer']}")

    print(f"Agent's Response: {response_from_agent}")
    print(f"The correct answer is: {question['Correct_Answer']}")

def save_chat_history(file_path, message):
    """Appends a message to the chat history file."""
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def main():
    # Define chat history file paths for practice and exam phases
    practice_chat_history_path = "practice_chat_history.txt"
    exam_chat_history_path = "exam_chat_history.txt"

    faiss_index = load_faiss_index(DB_FAISS_PATH)

    # Practice phase
    print("Starting practice phase...")
    for practice_question_id in range(1, 6):  # Practice questions IDs, currently low for testing purposes
        question = fetch_question_from_api(practice_question_id)
        agent_interaction(question, practice_chat_history_path, lcpp_llm, faiss_index)
        print(f"Completed practice question ID: {practice_question_id}")
        print("-" * 30)

    # Exam phase
    print("Starting exam phase...")
    exam_questions = generate_exam(10)  # Generate an exam, currently short for testing purposes
    for question in exam_questions:
        agent_interaction(question, exam_chat_history_path, lcpp_llm, faiss_index)
        print("-" * 30)

    print("Practice and exam phases completed.")

    while True:
        print("Sleep 5 min")
        time.sleep(300)


if __name__ == "__main__":
    main()
