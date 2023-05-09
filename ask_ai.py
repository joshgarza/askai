from llama_index import (
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    GPTListIndex,
    readers,
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display
from dotenv import load_dotenv
import random

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 500
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    # define LLM
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs
        )
    )

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )

    # index.save_to_disk('index.json')
    index.storage_context.persist(persist_dir="index.json")

    return index


def ask_ai(query):
    # index = GPTVectorStoreIndex.load_from_disk('index.json')
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="index.json")

    engineered_prompt = "You are a chatbot for Josh, JoshGPT. Craft a two sentence response to the query that is fun and casual. It MUST be a positive but realistic response. The following is the query:"
    # load index
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(engineered_prompt + query)

    return response.response

    # prompt_response = "Rewrite the following query in a way that abides by the following: a fun, casual response. Query: "

    # prompt_response_query = query_engine.query(prompt_response + response.response)

    # return prompt_response_query.response

    # slang_response = (
    #     "Rewrite the following query using as many words from slang_dict as possible: "
    # )

    # slang_response_query = query_engine.query(
    #     slang_response + prompt_response_query.response
    # )
    # slang_response2 = (
    #     "Rewrite the following query using as many words from slang_dict as possible: "
    # )

    # slang_response_query2 = query_engine.query(
    #     slang_response2 + slang_response_query.response
    # )

    # return slang_response_query2.response
