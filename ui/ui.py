import redis
import time
import os
import streamlit as st
from dotenv import load_dotenv
from rich import print
from redis.commands.search.query import Query
import openai
import numpy as np
from typing import List, Dict
from transformers import BartTokenizer, BartForConditionalGeneration


load_dotenv()
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_TEXT_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "idx:blogs"
KEY_PREFIX = "streamlit"
CHAT_HISTORY = KEY_PREFIX + ":" + "chat:history"
SUMMARY = KEY_PREFIX + ":" + "chat:summarize"


# Common Functions
def get_redis_conn() -> redis.Redis:
    redis_host, redis_port, redis_user, redis_pass = (
        os.getenv("redis_host"),
        os.getenv("redis_port"),
        os.getenv("redis_user"),
        os.getenv("redis_pass"),
    )
    if not redis_user:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    else:
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            username=redis_user,
            password=redis_pass,
            decode_responses=True,
        )
    return r


def clear_chat_history(r: redis.Redis, stream):
    if r.exists(stream):
        r.delete(stream)


def render_chat_history(r: redis.Redis, stream):
    if r.exists(stream):
        chat_history_msgs = r.xrange(stream)
        for ts, message in chat_history_msgs:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.markdown("Hi, How are you doing?", unsafe_allow_html=True)


def find_docs(query_vector):
    """
    Finds 3 similar docs in redis based on user prompt using vector similarity search
    """
    responses = []
    query = (
        Query("(*)=>[KNN 3 @vector $query_vector AS vector_score]")
        .sort_by("vector_score")
        .return_fields("vector_score", "id", "url", "author", "date", "title", "text")
        .dialect(2)
    )

    query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    result_docs = (
        r.ft(INDEX_NAME).search(query, {"query_vector": query_vector_bytes}).docs
    )
    for doc in result_docs:
        vector_score = round(1 - float(doc.vector_score), 2)
        if vector_score > 0.85:
            responses.append(
                {
                    "title": doc.title,
                    "url": doc.url,
                    "author": doc.author,
                    "date": doc.date,
                    "text": doc.text,
                }
            )
        print(vector_score)
    return responses


def get_embedding(doc):
    openai.api_key = OPENAI_API_KEY
    response = openai.Embedding.create(
        input=doc, model=OPENAI_EMBEDDING_MODEL, encoding_format="float"
    )
    embedding = response["data"][0]["embedding"]
    return embedding

def get_summary(text: List[str]) -> List[str]:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    input_ids = tokenizer.batch_encode_plus(
        text, truncation=True, padding=True, return_tensors="pt", max_length=1024
    )["input_ids"]
    summary_ids = model.generate(input_ids, max_length=500)
    summaries = [
        tokenizer.decode(
            s, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for s in summary_ids
    ]
    return summaries

def format_response(responses: List[Dict], summarize=False):
    formatted_response = ""
    for doc in responses:
        formatted_response += f"**[{doc['title']}]({doc['url']})**<br>"
        formatted_response += f"{doc['author']}<br>"
        formatted_response += f"{doc['date']}<br>"
        if summarize:
            formatted_response += f"*{get_summary(doc['text'])}*<br>"
        formatted_response += "<br>"
    return formatted_response


def get_response(r: redis.Redis, user_prompt: str, summarize=False):
    responses = find_docs(query_vector=get_embedding(user_prompt))
    formatted_response = (
        format_response(responses=responses, summarize=summarize)
        if responses
        else False
    )
    return formatted_response


def format_prompt(user_prompt: str):
    return user_prompt


# UI Rendering
r = get_redis_conn()


st.header("My Awesome Chatbot 💬", divider="rainbow")
# Initialize chat history
render_chat_history(r, stream=CHAT_HISTORY)

with st.sidebar:
    st.button(
        "Clear Chat History",
        type="primary",
        on_click=clear_chat_history,
        kwargs={"r": r, "stream": CHAT_HISTORY},
    )
    summarize = st.toggle(
        "Show me a summary", value=True if r.exists(SUMMARY) else False
    )

if summarize:
    r.set(SUMMARY, 0)
    with st.chat_message("assistant"):
        st.markdown(
            """Blogs will be summarized using 
            [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) model.
            Semantic Cache is enabled!"""
        )
else:
    with st.chat_message("assistant"):
        st.markdown("Summarization Disabled. Semantic Cache preserved for future use.")
        r.delete(SUMMARY)


prompt = st.chat_input("Ask me anything!")
if prompt:
    with st.chat_message("user"):
        fmt_prompt = format_prompt(prompt)
        st.markdown(fmt_prompt, unsafe_allow_html=True)
        r.xadd(CHAT_HISTORY, {"role": "user", "content": fmt_prompt})
    # Display response in chat message container
    with st.chat_message("assistant"):
        response = get_response(r, fmt_prompt, summarize=summarize)
        if not response:
            response = "I do not have enough information to answer this question"

        st.markdown(response, unsafe_allow_html=True)
        r.xadd(CHAT_HISTORY, {"role": "assistant", "content": response})
