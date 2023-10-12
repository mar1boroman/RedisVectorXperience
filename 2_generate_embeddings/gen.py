import os
import csv
import time
import openai
import configparser
from typing import List
from dotenv import load_dotenv
from rich import print


# from sentence_transformers import SentenceTransformer

load_dotenv()
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_TEXT_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_embedding(doc):
    openai.api_key = OPENAI_API_KEY
    response = openai.Embedding.create(
        input=doc, model=OPENAI_EMBEDDING_MODEL, encoding_format="float"
    )
    embedding = response["data"][0]["embedding"]
    return embedding


def create_embeddings(filepath, from_id=0, to_id=5):
    print(f"Using the input file {filepath} to generate embeddings...")
    records = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if i > to_id:
                break

            if from_id <= i <= to_id:
                doc = f"Title : {row[2]}, written by the {row[4]} on {row[3]}\n{row[5]}\nReference link: {row[1]}"
                row = {
                    "id": int(row[0]),
                    "url": row[1],
                    "title": row[2],
                    "date": row[3],
                    "author": row[4],
                    "text": row[5],
                    "embedding": get_embedding(doc),
                }
                records.append(row)

    return records


def save_embeddings(filepath, dict_list):
    fieldnames = dict_list[0].keys()

    with open(filepath, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dict_list:
            writer.writerow(row)


def main():
    st = time.time()
    records = create_embeddings(
        filepath="1_private_docs/redis_blogs.csv", from_id=0, to_id=1000
    )
    save_embeddings(
        filepath="2_generate_embeddings/redis_blogs_with_embeddings.csv",
        dict_list=records,
    )
    print(f"Embeddings generated ( {round(time.time() - st,2)} seconds )")


if __name__ == "__main__":
    main()
