from sentence_transformers import SentenceTransformer
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.llms import LlamaCpp
import fitz
import numpy as np
import time
import os
import torch
import faiss
import requests
import random
from llama_cpp import Llama

torch.cuda.empty_cache()


# Models for GPU
# model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGML"
# model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
#
# model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGML"
# model_basename = "llama-2-7b-chat.ggmlv3.q8_0.bin"
#
#
# model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
# model_basename = "llama-2-13b-chat.ggmlv3.q5_0.bin"
#
# model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
# model_basename = "llama-2-13b-chat.ggmlv3.q4_0.bin"

# CPU model
model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGUF"
model_basename = "llama-2-7b-chat.Q4_K_M.gguf"  # the model is in bin format

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


# Function to extract text from PDF files
def extract_text_from_pdf(pdf_folder):
    texts = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            with fitz.open(pdf_path) as pdf_doc:
                text = ""
                for page in pdf_doc:
                    text += page.get_text()
                texts.append(text)
    return texts


pdf_texts = extract_text_from_pdf("/usr/src/app/data/quality-dataset/")

# Get embeddings for the PDF texts
pdf_embeddings = get_embeddings(pdf_texts)

# Initialize a new Faiss index
pdf_index = faiss.IndexFlatL2(embeddings.shape[1])

# Convert embeddings to float32 if necessary
if pdf_embeddings.dtype != np.float32:
    pdf_embeddings = pdf_embeddings.astype(np.float32)

# Add PDF embeddings to the index
pdf_index.add(pdf_embeddings)

# Save the PDF Faiss index to disk
faiss.write_index(pdf_index, "pdf_faiss_index.index")


def load_faiss_index(index_path):
    return faiss.read_index(index_path)


def fetch_question_from_api(question_id):
    """Fetch a question from the API."""
    api_url = f'http://18.222.223.225:8000/questions/{question_id}'
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
    return exam  # Returns a list of question dictionaries


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


def generate_response_pdf(prompt):
    """Generate a response using the specified agent."""
    max_tokens_per_segment = 4096  # Maximum number of tokens per segment
    segments = [prompt[i:i+max_tokens_per_segment] for i in range(0, len(prompt), max_tokens_per_segment)]
    responses = []
    for segment in segments:
        response = lcpp_llm(
            prompt=segment,
            max_tokens=1000,
            temperature=0.5,
            top_p=0.95,
            repeat_penalty=1.2,
            top_k=50,
            stop=['USER:', 'ASSISTANT:'],
            echo=True
        )
        responses.append(response["choices"][0]["text"])
    return ' '.join(responses)


def agent_interaction(question, file_path, encoding_model, index, question_id):
    """
    Simulate the interaction based on a question object and store the chat history in the specified file.
    If an index is provided, perform a similarity search to assist with answering the question.
    """
    question_text = question['Question']
    choices_text = f" {question['Choice_A']}  {question['Choice_B']}  {question['Choice_C']}  {question['Choice_D']}  {question['Choice_E']}"

    prompt_for_agent = f"Question: {question_text} Choices: {choices_text} + Only one answer is correct. Answer only with the letter of the correct answer. \nAnswer:"

    # Perform similarity search if an index is provided
    if index is not None:
        question_embedding = encoding_model.encode([question_text], convert_to_tensor=True).cpu().numpy()
        D, I = index.search(question_embedding, k=1)

    response_from_agent = generate_response(prompt_for_agent)

    # Save the interaction to the specified chat history file
    save_chat_history(file_path, f"Question ID: {question_id}")
    save_chat_history(file_path, f"{response_from_agent}")
    save_chat_history(file_path, f"Correct Answer: {question['Correct_Answer']}")
    save_chat_history(file_path, f"-" * 60)

    print(f"{response_from_agent}")
    print(f"Correct Answer: {question['Correct_Answer']}")

    # print(f"{prompt_for_agent}")
    # print(f"{response_from_agent}")
    # print(f"The correct answer is: {question['Correct_Answer']}")

    print(f"{response_from_agent}")
    print(f"Correct Answer: {question['Correct_Answer']}")


def save_chat_history(file_path, message):
    """Appends a message to the chat history file."""
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def save_chat_history_pdf(file_path, messages):
    """Appends messages to the chat history file."""
    with open(file_path, "a", encoding="utf-8") as file:
        for message in messages:
            file.write(str(message) + "\n")  # Ensure message is converted to string


def main():
    practice_chat_history_path = "01-02-03-practice_chat_history.txt"
    exam_chat_history_path = "exam_chat_history.txt"
    api_exam_results_path = "01-02-03-04-api_exam_results.txt"
    pdf_exam_results_path = "01-02-03-pdf_exam_results.txt"
    quality_chat_history_path = "01-02-quality_conversation.txt"

    faiss_index = load_faiss_index(DB_FAISS_PATH)

    # Practice phase for API questions
    print("Starting practice phase for API questions...")
    api_practice_start_time = time.time()
    for practice_question_id in range(1, 490):  # Assuming these are placeholder values
        question = fetch_question_from_api(practice_question_id)
        agent_interaction(question, practice_chat_history_path, model, lcpp_llm, None, practice_question_id)
        print(f"Completed practice question ID: {practice_question_id}")
        print("-" * 30)
    api_practice_end_time = time.time()
    with open(practice_chat_history_path, "a", encoding="utf-8") as file:
        file.write(f"Total Practice Phase Time: {api_practice_end_time - api_practice_start_time} seconds\n")

    # Generate the same exam for both groups
    exam_questions = generate_exam(25)  # Adjust the number of questions as necessary

    # Exam phase for API questions
    print("Starting exam phase for API questions...")
    api_exam_start_time = time.time()
    for question in exam_questions:
        agent_interaction(question, exam_chat_history_path, model, faiss_index, question['ID'])
    api_exam_end_time = time.time()
    with open(api_exam_results_path, "a", encoding="utf-8") as file:
        file.write(f"Total Exam Time: {api_exam_end_time - api_exam_start_time} seconds\n")

    # Practice phase for PDF data
    print("Starting practice phase for PDF data...")
    pdf_practice_start_time = time.time()
    for _ in range(30):
        pdf_text = random.choice(pdf_texts)
        agent1_prompt = f"Discussion based on PDF content: {pdf_text[:100]}..."  # Shorten for brevity
        agent1_response = generate_response_pdf(agent1_prompt)
        save_chat_history_pdf(quality_chat_history_path, [agent1_prompt, agent1_response, "-" * 60])
    pdf_practice_end_time = time.time()
    with open(quality_chat_history_path, "a", encoding="utf-8") as file:
        file.write(f"Total Practice Phase Time for PDF: {pdf_practice_end_time - pdf_practice_start_time} seconds\n")

    # Exam phase for PDF data, using the same exam questions
    print("Starting exam phase for PDF data...")
    pdf_exam_start_time = time.time()
    for question in exam_questions:
        agent_interaction(question, pdf_exam_results_path, model, pdf_index,
                          question['ID'])  # Note the use of pdf_index if relevant
    pdf_exam_end_time = time.time()
    with open(pdf_exam_results_path, "a", encoding="utf-8") as file:
        file.write(f"Total Exam Time for PDF: {pdf_exam_end_time - pdf_exam_start_time} seconds\n")

    print("Practice and exam phases for both groups completed.")

    while True:
        print("Sleep 5 min")
        time.sleep(300)


if __name__ == "__main__":
    main()
