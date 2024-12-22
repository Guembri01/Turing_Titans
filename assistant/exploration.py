from pathlib import Path
import chromadb
import pandas as pd
from assistant import get_chromadb


def _assert_collection_exists(db_directory: Path, db_collection: str) -> None:
    if not db_directory.is_dir():
        raise NotADirectoryError(f"No vectorstore found at '{db_directory}'.")
    try:
        client_settings = chromadb.config.Settings(
            is_persistent=True, persist_directory=str(db_directory)
        )
        client = chromadb.Client(client_settings)
    except Exception as e:
        raise RuntimeError(f"Cannot open vectorstore at '{db_directory}'.") from e
    else:
        collection_names = [collection.name for collection in client.list_collections()]
        if db_collection not in collection_names:
            raise RuntimeError(
                f"Collection '{db_collection}' doesn't exist in the "
                f"vectorstore at '{db_directory}'."
            )


def get_docs_df(db_directory: Path, db_collection: str) -> pd.DataFrame:
    try:
        _assert_collection_exists(db_directory, db_collection)
    except Exception:
        return pd.DataFrame(columns=["id", "source", "page", "document", "embedding"])
    
    vectorstore = get_chromadb(persist_directory=db_directory, collection_name=db_collection)
    response = vectorstore.get(include=["metadatas", "documents", "embeddings"])
    
    # Extract the data
    ids = response.get("ids", [])
    sources = [metadata.get("source") for metadata in response.get("metadatas", [])]
    pages = [metadata.get("page", -1) for metadata in response.get("metadatas", [])]
    
    # Ensure no None values in documents
    documents = [doc if doc is not None else "" for doc in response.get("documents", [])]  # Replace None with empty string
    
    embeddings = response.get("embeddings", [])
    
    # Ensure all arrays are of the same length
    max_len = max(len(ids), len(sources), len(pages), len(documents), len(embeddings))
    
    # Pad shorter lists with None or appropriate default values
    ids = ids[:max_len] + [None] * (max_len - len(ids))
    sources = sources[:max_len] + [None] * (max_len - len(sources))
    pages = pages[:max_len] + [-1] * (max_len - len(pages))
    documents = documents[:max_len] + [None] * (max_len - len(documents))
    embeddings = embeddings[:max_len] + [None] * (max_len - len(embeddings))

    # Create the DataFrame
    docs_data = {
        "id": ids,
        "source": sources,
        "page": pages,
        "document": documents,
        "embedding": embeddings
    }

    return pd.DataFrame(docs_data)


def get_questions_df(db_directory: Path, db_collection: str) -> pd.DataFrame:
    try:
        _assert_collection_exists(db_directory, db_collection)
    except Exception:
        return pd.DataFrame(columns=["id", "question", "answer", "sources", "embedding"])

    vectorstore = get_chromadb(persist_directory=db_directory, collection_name=db_collection)
    response = vectorstore.get(include=["metadatas", "documents", "embeddings"])

    # Extract the data
    ids = response.get("ids", [])
    
    # Ensure no None values in questions (page_content)
    questions = [question if question is not None else "" for question in response.get("documents", [])]  # Replace None with empty string
    answers = [metadata.get("answer") for metadata in response.get("metadatas", [])]
    sources = [
        metadata.get("sources", "").split(",") for metadata in response.get("metadatas", [])
    ]
    embeddings = response.get("embeddings", [])

    # Ensure all arrays are of the same length
    max_len = max(len(ids), len(questions), len(answers), len(sources), len(embeddings))

    # Pad shorter lists with None or appropriate values
    ids = ids[:max_len] + [None] * (max_len - len(ids))
    questions = questions[:max_len] + [None] * (max_len - len(questions))
    answers = answers[:max_len] + [None] * (max_len - len(answers))
    sources = sources[:max_len] + [[]] * (max_len - len(sources))  # Empty lists for sources
    embeddings = embeddings[:max_len] + [None] * (max_len - len(embeddings))

    # Create the DataFrame
    questions_data = {
        "id": ids,
        "question": questions,
        "answer": answers,
        "sources": sources,
        "embedding": embeddings
    }

    return pd.DataFrame(questions_data)

def get_docs_questions_df(
    docs_db_directory: Path,
    docs_db_collection: str,
    questions_db_directory: Path,
    questions_db_collection: str,
) -> pd.DataFrame:
    # Get the documents and questions DataFrames
    docs_df = get_docs_df(docs_db_directory, docs_db_collection)
    docs_df["type"] = "doc"
    
    questions_df = get_questions_df(questions_db_directory, questions_db_collection)
    questions_df["type"] = "question"

    # Process the questions dataframe to add useful information
    questions_df["num_sources"] = questions_df["sources"].apply(len)
    questions_df["first_source"] = questions_df["sources"].apply(
        lambda x: next(iter(x), None)
    )

    # Process the documents dataframe to associate them with the questions
    if len(questions_df):
        docs_df["used_by_questions"] = docs_df["id"].apply(
            lambda doc_id: questions_df[
                questions_df["sources"].apply(lambda sources: doc_id in sources)
            ]["id"].tolist()
        )
    else:
        docs_df["used_by_questions"] = [[] for _ in range(len(docs_df))]

    docs_df["used_by_num_questions"] = docs_df["used_by_questions"].apply(len)
    docs_df["used_by_question_first"] = docs_df["used_by_questions"].apply(
        lambda x: next(iter(x), None)
    )

    # Combine docs and questions data into one dataframe
    df = pd.concat([docs_df, questions_df], ignore_index=True)
    return df
