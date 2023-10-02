import redis
import time
import streamlit as st


# Common Functions
def get_redis_conn():
    import configparser

    configparser = configparser.ConfigParser()
    configparser.read("config.ini")
    redis_host, redis_port, redis_user, redis_pass = (
        configparser["Redis"]["redis_host"],
        configparser["Redis"]["redis_port"],
        configparser["Redis"]["redis_user"],
        configparser["Redis"]["redis_pass"],
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


def clear_chat_history(stream):
    if r.exists(stream):
        r.delete(stream)


def render_chat_history(stream):
    if r.exists(stream):
        chat_history_msgs = r.xrange(stream)
        for ts, message in chat_history_msgs:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown("Hi, How are you doing?")


def stream_response(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.15)


def get_response(user_prompt: str):
    return user_prompt

def format_prompt(user_prompt:str):
    return user_prompt

# UI Rendering
r = get_redis_conn()
ui_prefix = "streamlit"
chat_history_stream = ui_prefix + ":" + "chat:history"


st.header("My Awesome Chatbot 💬", divider="rainbow")
# Initialize chat history
render_chat_history(stream=chat_history_stream)

with st.sidebar:
    st.button(
        "Clear Chat History",
        type="primary",
        on_click=clear_chat_history,
        kwargs={"stream": chat_history_stream},
    )


with st.container():
    # Accept user input
    prompt = st.chat_input("Ask me anything!")
    if prompt:
        with st.chat_message("user"):
            fmt_prompt = format_prompt(prompt)
            st.markdown(fmt_prompt)
            r.xadd(chat_history_stream, {"role": "user", "content": fmt_prompt})
        # Display response in chat message container
        with st.chat_message("assistant"):
            response = get_response(fmt_prompt)
            st.markdown(response)
            r.xadd(chat_history_stream, {"role": "assistant", "content": response})
