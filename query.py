import chromadb
import os
from openai import OpenAI
import argparse

# os.environ["OPENAI_API_KEY"] = Here you need to put your own API key


def query_db(query_text, n_results=5):
    """
    Queries the ChromaDB database for documents that are relevant to the query_text.

    Parameters:
    - query_text (str): The text to query in the database.
    - n_results (int): The number of results to return. Default is 5.

    Returns:
    - tuple: A tuple containing two lists: documents and titles. Each list contains strings.
    """
    collection_name = "medium_articles"

    chroma_client = chromadb.PersistentClient(path="medium_db")
    collection = chroma_client.get_collection(name=collection_name)

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=['documents', 'metadatas']
    )

    documents = [doc for doc in results['documents'][0]]
    titles = [meta['title'] for meta in results['metadatas'][0]]

    return documents, titles


def generate_answer_with_openai(question, n_results=5):
    """
    Generates an answer to a given question using documents fetched from ChromaDB and the OpenAI API.

    Parameters:
    - question (str): The question to generate an answer for.
    - n_results (int): The number of documents to fetch from the database for generating the answer. Default is 5.

    Returns:
    - str: The generated answer, with sources included if available.
    """
    documents, titles = query_db(question, n_results)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    articles = ""
    prompt = f"Question: {question}\n\n"
    for title, doc in zip(titles, documents):
        prompt += f"Article Title: {title}\n{doc}\n\n"
        articles += f"; {title}" if articles else f"Sources: {title}"
    prompt += "Based on the above articles, please answer the question."

    max_tokens = 4096
    prompt = prompt[:max_tokens]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a knowledgeable assistant who provides answers based on the provided articles."},
            {"role": "user", "content": prompt}
        ],
    )

    answer = response.choices[0].message.content + f"\n\n{articles}"
    return answer.strip()


def main():
    """
    Parses command-line arguments and generates an answer to the specified question.
    Optionally includes detailed information based on a flag.
    """
    parser = argparse.ArgumentParser(description='Generate answers using OpenAI and ChromaDB.')
    parser.add_argument('question', type=str, help='The question to generate an answer for.')
    parser.add_argument('--detailed', action='store_true',
                        help='Include detailed information in the output, such as each article title and content.')
    args = parser.parse_args()

    if args.detailed:
        documents, titles = query_db(args.question)
        print("Detailed Information:\n")
        for title, doc in zip(titles, documents):
            print(f"Article Title: {title}\n{doc}\n")
        print("Summary Answer:\n")

    print(generate_answer_with_openai(args.question))


if __name__ == "__main__":
    main()
