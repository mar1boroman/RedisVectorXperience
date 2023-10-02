import redis
import uuid
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
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType


load_dotenv()

OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_TEXT_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "idx:blogs"
KEY_PREFIX = "streamlit"
CHAT_HISTORY = KEY_PREFIX + ":" + "chat:history"
SUMMARY = KEY_PREFIX + ":" + "chat:summarize"
SEMANTIC_CACHE_PREFIX = KEY_PREFIX + ":" + "semantic_cache"
SEMANTIC_CACHE_INDEX_NAME = "idx:prompts"
openai.api_key = OPENAI_API_KEY


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
        print(doc.title, vector_score)
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


def check_semantic_cache(r, user_prompt_embedding):
    response = ""
    query_vector = user_prompt_embedding
    query = (
        Query("(*)=>[KNN 1 @vector $query_vector AS vector_score]")
        .sort_by("vector_score")
        .return_fields("vector_score", "response", "prompt")
        .dialect(2)
    )

    query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    result_docs = (
        r.ft(SEMANTIC_CACHE_INDEX_NAME)
        .search(query, {"query_vector": query_vector_bytes})
        .docs
    )
    for doc in result_docs:
        vector_score = round(1 - float(doc.vector_score), 2)
        if vector_score > 0.9:
            response = doc.response
        print(doc.prompt, vector_score)
    return response if response else False


def format_response(
    r: redis.Redis,
    user_prompt: str,
    user_prompt_embedding,
    responses,
    summarize=False,
):
    formatted_response = ""
    for doc in responses:
        formatted_response += f"**[{doc['title']}]({doc['url']})**<br>"
        formatted_response += f"{doc['author']}<br>"
        formatted_response += f"{doc['date']}<br>"
        if summarize:
            summary = get_summary([doc["text"]])[0]
            formatted_response += f"*{summary}*<br>"

        formatted_response += "<br>"

    if summarize:
        semantic_cache_keyname = (
            SEMANTIC_CACHE_PREFIX + ":" + str(uuid.uuid4()).replace("-", "")
        )
        r.json().set(
            name=semantic_cache_keyname,
            path="$",
            obj={
                "prompt": user_prompt,
                "response": formatted_response,
                "prompt_embedding": user_prompt_embedding,
            },
        )

    return formatted_response


def get_response(r: redis.Redis, user_prompt: str, summarize=False):
    query_vector = get_embedding(user_prompt)
    if summarize:
        cached_response = check_semantic_cache(r, user_prompt_embedding=query_vector)
        if cached_response:
            return "Cached Response<br>" + cached_response

    responses = find_docs(query_vector=query_vector)
    formatted_response = (
        format_response(
            r=r,
            user_prompt=user_prompt,
            user_prompt_embedding=query_vector,
            responses=responses,
            summarize=summarize,
        )
        if responses
        else False
    )
    return formatted_response


def create_semantic_search_index():
    # if SEMANTIC_CACHE_INDEX_NAME in r.execute_command("FT._LIST"):
    #     print("Dropping the existing Index")
    #     r.ft(SEMANTIC_CACHE_INDEX_NAME).dropindex()

    if SEMANTIC_CACHE_INDEX_NAME not in r.execute_command("FT._LIST"):
        idx_schema = (
            TextField(
                "$.prompt",
                as_name="prompt",
            ),
            TextField(
                "$.response",
                as_name="response",
            ),
            VectorField(
                "$.prompt_embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": 1536,
                    "DISTANCE_METRIC": "COSINE",
                },
                as_name="vector",
            ),
        )

        idx_definition = IndexDefinition(
            prefix=[SEMANTIC_CACHE_PREFIX], index_type=IndexType.JSON
        )
        res = r.ft(SEMANTIC_CACHE_INDEX_NAME).create_index(
            fields=idx_schema, definition=idx_definition
        )

        if res == "OK":
            print(f"Index {SEMANTIC_CACHE_INDEX_NAME} created successfully")
            percent_indexed = 0.0
            idx_info = r.ft(SEMANTIC_CACHE_INDEX_NAME).info()
            percent_indexed, num_docs, hash_indexing_failures = (
                idx_info["percent_indexed"],
                idx_info["num_docs"],
                idx_info["hash_indexing_failures"],
            )
            time.sleep(2)
            while float(percent_indexed) < 1:
                time.sleep(5)
                idx_info = r.ft(SEMANTIC_CACHE_INDEX_NAME).info()
                percent_indexed, num_docs, hash_indexing_failures = (
                    idx_info["percent_indexed"],
                    idx_info["num_docs"],
                    idx_info["hash_indexing_failures"],
                )
            print(
                f"{round(float(percent_indexed)*100,2)} % indexed, {num_docs} documents indexed, {hash_indexing_failures} documents indexing failed"
            )
            print("Index ready!")


def get_context(user_prompt):
    if not rag:
        return user_prompt
    else:
        query_vector = get_embedding(user_prompt)
        responses = find_docs(query_vector=query_vector)
        context = ""
        for doc in responses:
            context += doc["text"]
            context += "\n"
        return (
            f"""Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, say that you don't know, don't try to make up an answer.
        Context:
        {context}
        Question:
        {user_prompt}
        """
            if context
            else False
        )


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

    chat = st.toggle("Chat with OpenAI")

    rag = st.toggle("Use RAG Model", disabled=False if chat else True)

    if not chat:
        rag = False


if summarize:
    create_semantic_search_index()
    r.set(SUMMARY, 0)
    with st.chat_message("assistant"):
        print(
            """Blogs will be summarized using 
            [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) model.
            Semantic Cache is enabled!"""
        )
else:
    with st.chat_message("assistant"):
        print("Summarization Disabled. Semantic Cache preserved for future use.")
        r.delete(SUMMARY)


prompt = st.chat_input("Ask me anything!")


if chat:
    create_semantic_search_index()

    print("Enabled chat with Open AI")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
            
            enhanced_prompt = prompt
            if rag:
                print("Enabled RAG Model")
                enhanced_prompt = get_context(prompt)
            else:
                print("Running without RAG Model")
                enhanced_prompt = (
                    prompt
                    + "<br>If you don't know the answer, say that you don't know, don't try to make up an answer."
                )
            r.xadd(CHAT_HISTORY, {"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            user_prompt_embedding = get_embedding(prompt)

            message_placeholder = st.empty()
            full_response = ""

            cached_response = check_semantic_cache(
                r, user_prompt_embedding=user_prompt_embedding
            )
            if cached_response:
                full_response = cached_response
                message_placeholder.markdown("Cached Response : " + cached_response)
            else:
                for response in openai.ChatCompletion.create(
                    model=OPENAI_TEXT_MODEL,
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                if rag:
                    semantic_cache_keyname = (
                        SEMANTIC_CACHE_PREFIX + ":" + str(uuid.uuid4()).replace("-", "")
                    )
                    r.json().set(
                        name=semantic_cache_keyname,
                        path="$",
                        obj={
                            "prompt": prompt,
                            "response": full_response,
                            "prompt_embedding": user_prompt_embedding,
                        },
                    )
            r.xadd(CHAT_HISTORY, {"role": "assistant", "content": full_response})
else:
    print("Now only working with local documents")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=True)
            r.xadd(CHAT_HISTORY, {"role": "user", "content": prompt})
        # Display response in chat message container
        with st.chat_message("assistant"):
            response = get_response(r, prompt, summarize=summarize)
            if not response:
                response = "I do not have enough information to answer this question"

            st.markdown(response, unsafe_allow_html=True)
            r.xadd(CHAT_HISTORY, {"role": "assistant", "content": response})
