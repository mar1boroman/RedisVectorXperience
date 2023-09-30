import configparser
from typing import List
import streamlit as st
import time
import os
import asyncio
import configparser
import pandas as pd
from redisvl.vectorize.text import HFTextVectorizer
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from transformers import BartTokenizer, BartForConditionalGeneration
from redisvl.llmcache.semantic import SemanticCache

# Global variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf = HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
index_schema_file = "blog_index.yaml"
vector_field_name = "blog_embedding"
index = AsyncSearchIndex.from_yaml(index_schema_file)


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


def find_blogs(query_string, index, vector_field_name, num_docs=3):
    # use the HuggingFace vectorizer again to create a query embedding
    query_embedding = hf.embed(query_string)
    query = VectorQuery(
        vector=query_embedding,
        vector_field_name=vector_field_name,
        return_fields=["url", "title", "date", "author", "text"],
        num_results=num_docs,
    )

    return index.search(query.query, query_params=query.params)


def get_responses(results):
    responses = []
    for doc in results.docs:
        responses.append([doc.title, doc.date, doc.author, doc.url, doc.text])

    return responses


# Building a streamlit chatbot

index.connect(get_redis_uri())
st.title("Blog Recommendation Engine")
summarize_flag = st.checkbox("Auto-Summarize Blogs (Use LLM Semantic Cache)")
cache = SemanticCache(redis_url=get_redis_uri(), threshold=0.6)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)


if prompt := st.chat_input("What do you want to know about Redis Enterprise ?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)
        cached_result = cache.check(prompt=prompt) if summarize_flag else ''

    start = time.time()
    
    full_response = ""
    
    if cached_result:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response += cached_result[0]
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            results = asyncio.run(
                find_blogs(
                    query_string=prompt,
                    index=index,
                    vector_field_name=vector_field_name,
                    num_docs=3,
                )
            )
            for response in get_responses(results):
                title, date, author, url, text = response
                full_response += f"""
                **Blog Title: {title}**
                <br>
                _{author}, {date}_
                <br>
                {url}
                """
                if summarize_flag:
                    full_response += f"<br>**Summary**<br>{old_summarization_pipeline([text])[0]}<br><br>"

    
    time_taken_str = (
        f"<br>:yellow[_Time taken for the response: {time.time() - start}_]"
    )
    
    is_semantic_cache_enabled = '' if summarize_flag else '_This query did not use LLM Semantic Cache._'

    message_placeholder.markdown(full_response + time_taken_str + is_semantic_cache_enabled, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response + time_taken_str + is_semantic_cache_enabled})
    
    if summarize_flag:
        cache.store(prompt, full_response)
