import logging
import os
import sys
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

# Check that OPENAI_API_KEY is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Define which LLM we are going to use for response generation
# and which embedding model to create embeddings. Default is
# GPT-3 Davinci and OpenAI embeddings model ADA
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
embed_model = OpenAIEmbedding()

# Uncomment to see debug logs #
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def load_data():
    reader = SimpleDirectoryReader(input_dir="./documents", recursive=True)
    # creating Callback manager object to show each sub process details \
    callback_manager = CallbackManager([llama_debug])
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(
        docs, embed_model=embed_model, callback_manager=callback_manager
    )
    return index


def create_custom_chatEngine(index):
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    # list of `ChatMessage` objects
    context_prompt = (
        "You are a chatbot helping red cross volunteers for the 2024 Olympic Games."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
        "\n Please answer the question always in the language of the quetsion. Always stay polite and share your source."
    )

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        context_prompt=context_prompt,
        verbose=True,
    )
    return chat_engine


def insertTable(session_id, text):
    print(
        supabase.table("chatlog")
        .upsert({"session_id": session_id, "history": text})
        .execute()
    )


# Define your supabase key and url
# For security reasons please create a enviroment variable for it
supabase_key = os.environ["DB_KEY"] = "PASTE_YOUR_SUPABASE_DB_KEY_HERE"
supabase_url = "PASTE_YOUR_URL_HERE"

# TODO - connect to supabase
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


with gr.Blocks(fill_height=True) as demo:
    chat_engine = create_custom_chatEngine(index)

    # Create new session_id
    session_id = str(uuid.uuid1())

    chatbot = gr.Chatbot(scale=2)
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
    print("Ready to chat!")
    demo.queue().launch(share=False)
