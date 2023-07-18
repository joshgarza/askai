from llama_index import (
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)

# from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def build_service_context():
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

    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0.8, model_name="gpt-3.5-turbo", max_tokens=num_outputs
        )
    )

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    return service_context


def construct_index(directory_path):
    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=build_service_context()
    )

    index.storage_context.persist(persist_dir="index.json")

    return index


def ask_ai(query):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="index.json")

    # load index
    index = load_index_from_storage(
        storage_context, service_context=build_service_context()
    )

    query_engine = index.as_query_engine()
    engineered_prompt = "You are a chatbot for Josh, JoshGPT. Craft a two sentence response to the query that is fun and casual. It MUST be a positive but realistic response. The following is the query:"
    response = query_engine.query(engineered_prompt + query)

    return response.response
