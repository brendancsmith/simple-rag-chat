import chainlit as cl
import chromadb
import chromadb.errors
from chromadb.config import Settings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from documents import process_docs
from prompts import EXAMPLE_PROMPT, PROMPT, WELCOME_MESSAGE

CHAT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

def create_store(*, embedding: Embeddings) -> VectorStore:
    """Takes a list of Langchain Documents and an Langchain embeddings wrapper
    over encoder models, and index the data into a ChromaDB as a search engine

    Args:
        docs (List[Document]): list of documents to be ingested
        embeddings (Embeddings): encoder model

    Returns:
        VectorStore: vector store for RAG
    """

    # Initialize Chromadb client to enable resetting and disable telemtry
    client = chromadb.EphemeralClient()
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        anonymized_telemetry=False,
        persist_directory=".chromadb",
        allow_reset=True,
    )

    # Reset the search engine to ensure we don't use old copies.
    # NOTE: we do not need this for production
    store = Chroma(
        client=client,
        client_settings=client_settings
    )
    store._client.reset()

    # Initalize the VectorStore with the ChromaDB client and the embedding function
    store = Chroma(
        client=client,
        client_settings=client_settings,
        collection_name="default",
        embedding_function=embedding,
    )

    cl.user_session.set("store", store)
    return store


async def create_chain():
    """Creates and initializes a retrieval-augmented generation (RAG) chain for document processing.

    This function sets up an embedding model and a retrieval store, then constructs a chain that utilizes a language model for generating responses based on retrieved documents. It handles errors during the store creation and communicates any issues to the user.

    Args:
        None

    Returns:
        chain: The initialized RAG chain for document processing.
    """

    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    try:
        store = await cl.make_async(create_store)(embedding=embedding)
        cl.user_session.set("store", store)
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        raise RuntimeError from e

    # RAG Chain
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, streaming=True)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=store.as_retriever(max_tokens_limit=4097),
        chain_type_kwargs={"prompt": PROMPT, "document_prompt": EXAMPLE_PROMPT},
    )

    cl.user_session.set("chain", chain)
    return chain


async def add_documents(files):
    """Add documents to the search engine

    Args:
        files (List[AskFileResponse]: list of files to be ingested
    """

    docs = await process_docs(files)

    if not docs:
        return

    store = cl.user_session.get("store")
    if not isinstance(store, VectorStore) or store is None:
        raise TypeError("Store in user session is not a valid type")

    store.add_documents(docs)

    # add an action to upload more documents
    button = cl.Action(name="more_docs", value="", label="Add documents")
    msg = cl.Message(content="You can now ask questions!", actions=[button])
    await msg.send()


async def ask_for_docs(content: str):
    """Ask user to upload documents

    Args:
        content (str): message content to show to user

    Returns:
        List[cl.AskFileResponse]: list of files uploaded by user

    Raises:
        TypeError: when message content is empty
    """

    if not len(content):
        raise TypeError("AskFileMessage content cannot be empty")

    files = await cl.AskFileMessage(
        content=content,
        accept=["text/plain", "application/pdf"],
        max_size_mb=100,
        max_files=10,
    ).send()

    return files


@cl.action_callback(name="more_docs")
async def more_docs(action):
    await action.remove()

    files = await ask_for_docs("Add more documents")

    await add_documents(files)
    

@cl.on_chat_start
async def on_chat_start():
    """This function is run at every chat session starts to ask user for file,
    index it, and build the RAG chain.

    Raises:
        RuntimeError: when there is an error in indexing the documents
    """

    await create_chain()

    # Asking user to to upload PDF or txt files to chat with
    files = None
    while files is None:
        files = await ask_for_docs(WELCOME_MESSAGE)

    await add_documents(files)


@cl.on_message
async def on_message(message: cl.Message):
    """Invoked whenever we receive a Chainlit message.

    Args:
        message (cl.Message): user input
    """

    chain = cl.user_session.get("chain")
    if type(chain) is not RetrievalQAWithSourcesChain:
        raise TypeError("Chain in user session is not a RetrievalQAWithSourcesChain")

    response = await chain.acall(
        message.content,
        callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)],
    )

    answer = response["answer"]
    sources = response["sources"].strip()
    source_elements = []

    store = cl.user_session.get("store")
    if not isinstance(store, VectorStore):
        raise TypeError("Store in user session is not a VectorStore")
    docs = store._collection.get() # type: ignore

    metadatas = docs["metadatas"]
    all_sources = [m["source"] for m in metadatas]

    # Adding sources to the answer
    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split("`, `"):
            source_name = source.strip().replace("`", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs["documents"][index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            formatted_sources = '\n- '.join(found_sources)
            answer += f"\n\nSources:\n- {formatted_sources}"
        else:
            answer += "\n\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()
