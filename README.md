# Article Retrieval System

Version: 1.0

## Description

This Python script queries a ChromaDB database containing the "1300+ Towards Data Science Medium Articles Dataset" for information relevant to a user-provided question. It then uses OpenAI's GPT-3.5 to generate an answer based on this information. The dataset includes titles and full text of articles from the "Towards Data Science" publication on Medium, supporting a range of NLP tasks.

Link to the dataset: https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset

## Usage

Install dependencies.

```python
pip install -r requirements.txt
```

Create and fill the Chroma DB.

```python
python create_database.py
```

Query the Chroma DB.

```python
python query.py "What is NLP?"
```

You can also include `--detailed` flag for more detailed information (chunks used as a source for the response will be printed out along with corresponding article titles).

```python
python query.py "What is NLP?" --detailed
```

## Author:
Michał Markowski
Contact at: michoch4@gmail.com
