import os
import time
import uuid

import gradio as gr
from llama_index.core import (
    PromptTemplate,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager, CBEventType, LlamaDebugHandler
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
llama_debug = LlamaDebugHandler(print_trace_on_end=True)

# Service context contains information about which LLM we
# are going to use for response generation and which
# embedding model to create embeddings. Default is
# GPT-3 Davinci and OpenAI embeddings model ADA

import logging
import sys

# Uncomment to see debug logs #
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def load_data():
    embed_model = OpenAIEmbedding()
    print("embed_model loaded")
    reader = SimpleDirectoryReader(
        input_dir="./documents", recursive=True
    )  # creating Callback manager object to show each sub process details \
    callback_manager = CallbackManager([llama_debug])
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(
        docs, embed_model=embed_model, callback_manager=callback_manager
    )
    return index


def create_custom_chatEngine(index):
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    # list of `ChatMessage` objects
    template = (
        "Following Informations : \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Please answer the question always in the language of the quetsion. Always stay polite and share your source."
    )
    # qa_template = PromptTemplate(template)

    # query_engine = index.as_query_engine(text_qa_template=qa_template, llm=llm)
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        llm=llm,
        context_prompt=(
            "You are a chatbot helping red cross volunteers for the 2024 Olympic Games."
            "Here are the relevant documents for the context:\n"
            "{context_str}"
            "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
            "\n Please answer the question always in the language of the quetsion. Always stay polite and share your source."
        ),
        verbose=True,
    )
    return chat_engine


def insertTable(sessionId, text):
    print(
        supabase.table("chatlog")
        .upsert({"session_id": sessionId, "history": text})
        .execute()
    )


# Add here your secret API-KEY from Supabase
# For security reasons please create a enviroment variable for it
key = os.environ["DB_KEY"] = "PASTE_YOUR_SUPABASE_DB_KEY_HERE"

# Replace with your OpenAI API key
# For security reasons please create a enviroment variable for it

# os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_OPENAI_KEY_HERE"

# Add here your supabase-URL
url = "PASTE_YOUR_URL_HERE"

supabase = None

# Load index from disk
try:
    storage_context = StorageContext.from_defaults(persist_dir="./index_files")
    print("loading index from files...")
    index = load_index_from_storage(storage_context)
except FileNotFoundError as e:
    # Create the index
    print("loading data")
    index = load_data()
    print("data loaded")
    # Persist index
    index.storage_context.persist("./index_files")

print("index is ready")
print(index)

with gr.Blocks() as demo:
    chat_engine = create_custom_chatEngine(index)

    # Create new session_id
    session_id = str(uuid.uuid1())

    chatbot = gr.Chatbot()
    msg = gr.Textbox(
        label="⏎ for sending",
        placeholder="Ask me something",
    )
    clear = gr.Button("Delete")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message = chat_engine.chat(user_message)
        history[-1][1] = ""
        for character in bot_message.response:
            history[-1][1] += character
            time.sleep(0.01)
            yield history
        # insert the chat-history to our table
        # insertTable(session_id,history)

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(
        bot, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=True)

if __name__ == "__main__":
    print("Hello World!")
    demo.queue().launch(share=False)
