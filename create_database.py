import sys
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import uuid


def chunk_text(text, chunk_size=512):
    """
    Splits a given text into smaller chunks of a specified size.

    Parameters:
    - text (str): The text to be chunked.
    - chunk_size (int, optional): The number of words per chunk. Defaults to 512.

    Returns:
    - list: A list of strings, where each string is a chunk of the original text.
    """
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def process_and_add_to_chromadb(input_file='data/medium.csv'):
    """
    Processes a CSV file containing textual data and adds it to a ChromaDB collection.

    This function performs several key operations:
    1. Reads a CSV file into a pandas DataFrame.
    2. Iterates through each row of the DataFrame, chunking the text from each row.
    3. Generates unique identifiers for each chunk.
    4. Creates a ChromaDB client and a collection if it doesn't exist.
    5. Uses an embedding function for the collection.
    6. Adds the chunks, metadata, and IDs to the ChromaDB collection.

    Parameters:
    - input_file (str, optional): The path to the CSV file to be processed. Defaults to 'data/medium.csv'.
    """
    df = pd.read_csv(input_file)

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        title = row['Title']
        text_chunks = chunk_text(row['Text'])

        for chunk in text_chunks:
            documents.append(chunk)
            metadatas.append({"title": title})
            ids.append(str(uuid.uuid4()))

    chroma_client = chromadb.PersistentClient(path="medium_db")

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    collection = chroma_client.get_or_create_collection(name="medium_articles",
                                                        embedding_function=sentence_transformer_ef)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    process_and_add_to_chromadb(input_file_path)
