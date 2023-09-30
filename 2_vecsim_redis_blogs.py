import os
import time
import asyncio
import configparser
import pandas as pd
from typing import List
from rich import print
from redisvl.vectorize.text import HFTextVectorizer
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from transformers import BartTokenizer, BartForConditionalGeneration
from redisvl.llmcache.semantic import SemanticCache


# Global variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf = HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def old_summarization_pipeline(text: List[str]) -> List[str]:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    input_ids = tokenizer.batch_encode_plus(
        text, truncation=True, padding=True, return_tensors="pt", max_length=1024
    )["input_ids"]
    summary_ids = model.generate(input_ids, max_length=300)
    summaries = [
        tokenizer.decode(
            s, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for s in summary_ids
    ]
    return summaries


def get_user_input():
    user_input = input("Enter your query (or q to quit): \n")
    return user_input


def get_redis_uri():
    parser = configparser.ConfigParser()
    parser.read("config.ini")
    return parser["RedisURI"]["uri"]


async def find_blogs(query_string, index, vector_field_name, num_docs=3):
    # use the HuggingFace vectorizer again to create a query embedding
    query_embedding = hf.embed(query_string)
    query = VectorQuery(
        vector=query_embedding,
        vector_field_name=vector_field_name,
        return_fields=["url", "title", "date", "author", "text"],
        num_results=num_docs,
    )

    return await index.search(query.query, query_params=query.params)


def get_responses(results):
    responses = []
    for doc in results.docs:
        responses.append([doc.title, doc.date, doc.author, doc.url, doc.text])

    return responses


async def main():
    index_schema_file = "blog_index.yaml"
    vector_field_name = "blog_embedding"
    index = AsyncSearchIndex.from_yaml(index_schema_file)
    index.connect(get_redis_uri())

    cache = SemanticCache(redis_url=get_redis_uri(), threshold=0.7)

    enable_summary = input(
        "Do you want to enable autosummarize using BART Summarizer? [y/n] : "
    )
    if enable_summary.lower() == "y":
        withsummary = True
    else:
        withsummary = False

    # QnA Loop

    query_string = get_user_input()
    while query_string != "q":
        start = time.time()

        cached_result = cache.check(prompt=query_string)
        
        if cached_result:
            print("Retrieving cached response...")
            full_response = cached_result[0]
        else:
            results = await find_blogs(
                query_string=query_string,
                index=index,
                vector_field_name=vector_field_name,
                num_docs=3,
            )

            full_response = "\n"
            for response in get_responses(results):
                title, date, author, url, text = response
                full_response += f"""Blog Title: {title}\n{author}, {date}\n{url}"""
                if withsummary:
                    full_response += f"Summary\n{old_summarization_pipeline([text])[0]}\n\n"

            cache.store(query_string, full_response)

        full_response += f"Time taken for the response: {time.time() - start}\n"

        print(full_response)

        query_string = get_user_input()


if __name__ == "__main__":
    asyncio.run(main=main())
