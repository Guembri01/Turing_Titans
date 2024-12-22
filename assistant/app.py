from typing import Any, Callable, Dict, List, Optional, Type, Union, get_args
import streamlit as st
import typer
from visualization import VisualizationAgent
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import re
from search_agent import SearchAgent
from assistant import (
    get_chromadb,
    get_embeddings_model,
    get_embeddings_model_config,
    get_llm,
    get_llm_config,
    get_rag_chain,
    get_retriever,
    question_as_doc,
)
from assistant.exploration import get_docs_questions_df
from assistant.settings import Settings, settings
from assistant.types import (
    AVATARS,
    LLM,
    MODEL_TYPES,
    PREDEFINED_RELEVANCE_SCORE_FNS,
    RETRIEVER_SEARCH_TYPES,
    Message,
    ModelType,
    NestedMessage,
    PredefinedRelevanceScoreFn,
    RelevanceScoreFn,
    RetrieverSearchType,
)
import cloudscraper
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = typer.Typer()

def hash_model(model: Union[Embeddings, LLM]) -> int:
    """Generate a hash for the given model."""
    if isinstance(model, Embeddings):
        name, model_type = get_embeddings_model_config(model)
    else:
        name, model_type = get_llm_config(model)
    return hash(name) ^ hash(model_type)

HASH_FUNCS: Dict[Union[str, Type], Callable[[Any], Any]] = {
    AzureOpenAIEmbeddings: hash_model,
    OpenAIEmbeddings: hash_model,
    HuggingFaceEmbeddings: hash_model,
    AzureChatOpenAI: hash_model,
    ChatOpenAI: hash_model,
    HuggingFacePipeline: hash_model,
}

_get_llm = st.cache_resource(max_entries=1, show_spinner=False)(get_llm)
_get_embeddings_model = st.cache_resource(max_entries=1, show_spinner=False)(get_embeddings_model)

@st.cache_resource(show_spinner=False, hash_funcs=HASH_FUNCS)
def _get_rag_chain(
    llm: LLM,
    relevance_score_fn: RelevanceScoreFn,
    k: int,
    search_type: RetrieverSearchType,
    score_threshold: float,
    fetch_k: int,
    lambda_mult: float,
    embeddings_model: Embeddings,
) -> Runnable:
    """Create and return the RAG chain."""
    vectorstore = get_chromadb(
        embeddings_model,
        settings.docs_db_directory,
        settings.docs_db_collection,
        relevance_score_fn,
    )
    retriever = get_retriever(
        vectorstore, k, search_type, score_threshold, fetch_k, lambda_mult
    )
    chain = get_rag_chain(retriever, llm)
    return chain

@st.cache_resource(show_spinner=False, hash_funcs=HASH_FUNCS)
def _get_questions_chromadb(embeddings_model: Embeddings) -> Chroma:
    """Create and return the Chroma vectorstore for questions."""
    vectorstore = get_chromadb(
        embeddings_model,
        settings.questions_db_directory,
        settings.questions_db_collection,
    )
    return vectorstore

@st.cache_resource(show_spinner=False)
def get_or_create_spotlight_viewer() -> Any:
    """Get or create a Renumics Spotlight viewer."""
    try:
        from renumics import spotlight
        from renumics.spotlight import dtypes as spotlight_dtypes
    except ImportError:
        return None
    viewers = spotlight.viewers()
    if viewers:
        for viewer in viewers[:-1]:
            viewer.close()
        return spotlight.viewers()[-1]
    return spotlight.show(
        port=8002,
        no_browser=True,
        dtype={
            "used_by_questions": spotlight_dtypes.SequenceDType(
                spotlight_dtypes.str_dtype
            )
        },
        wait=False,
    )

def st_settings(default_settings: Settings) -> None:
    """Display and handle settings in Streamlit."""
    st.header("Settings")
    st.subheader("LLM")
    st.selectbox(
        "Type",
        get_args(ModelType),
        get_args(ModelType).index(default_settings.llm_type),
        format_func=lambda x: MODEL_TYPES.get(x, x),
        key="llm_type",
    )
    st.text_input("Name", value=default_settings.llm_name, key="llm_name")
    with st.expander("Advanced"):
        st.subheader("Retriever")
        st.selectbox(
            "Relevance Score Function",
            get_args(PredefinedRelevanceScoreFn),
            get_args(PredefinedRelevanceScoreFn).index(
                default_settings.relevance_score_fn
            ),
            format_func=lambda x: PREDEFINED_RELEVANCE_SCORE_FNS.get(x, x),
            key="relevance_score_fn",
            help="Distance function in the embedding space",
        )
        k = st.slider(
            "k",
            1,
            max(100, default_settings.k + 20),
            default_settings.k,
            key="k",
            help="Amount of documents to return",
        )
        search_type = st.selectbox(
            "Search Type",
            get_args(RetrieverSearchType),
            get_args(RetrieverSearchType).index(default_settings.search_type),
            format_func=lambda x: RETRIEVER_SEARCH_TYPES.get(x, x),
            key="search_type",
            help="Type of search",
        )
        st.slider(
            "Score Threshold",
            0.0,
            1.0,
            default_settings.score_threshold,
            key="score_threshold",
            help="Minimum relevance threshold",
            disabled=search_type != "similarity_score_threshold",
        )
        st.slider(
            "Fetch k",
            k,
            max(200, k * 2),
            max(default_settings.fetch_k, k + 10),
            key="fetch_k",
            help="Amount of documents to pass to MMR",
            disabled=search_type != "mmr",
        )
        st.slider(
            "MMR λ",
            0.0,
            1.0,
            default_settings.lambda_mult,
            key="lambda_mult",
            help="Diversity of results returned by MMR. 1 for minimum diversity and 0 for maximum.",
            disabled=search_type != "mmr",
        )
        st.subheader("Embeddings Model")
        st.write(
            f"Be sure to replace the vectorstore at '{default_settings.docs_db_directory}' "
            f"with one indexed by the respective embeddings model and stored in "
            f"the '{default_settings.docs_db_collection}' collection."
        )
        st.selectbox(
            "Type",
            get_args(ModelType),
            get_args(ModelType).index(default_settings.embeddings_model_type),
            format_func=lambda x: MODEL_TYPES.get(x, x),
            key="embeddings_model_type",
        )
        st.text_input(
            "Name",
            value=default_settings.embeddings_model_name,
            key="embeddings_model_name",
        )

def st_chat_messages(messages: List[Message]) -> None:
    """Display chat messages in Streamlit."""
    for message in messages:
        with st.chat_message(message.role, avatar=AVATARS.get(message.role)):
            if isinstance(message, NestedMessage):
                with st.expander(message.content):
                    for content in message.subcontents:
                        st.write(content)
            else:
                st.write(message.content)

def st_chat(on_question: Callable[[str], List[Message]]) -> None:
    """Handle chat interactions in Streamlit."""
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [Message("assistant", "Ask me a question")]

    if question := st.chat_input("Your question"):
        st.session_state.messages.append(Message("user", question))

    st_chat_messages(st.session_state.messages)

    if st.session_state.messages[-1].role == "user":
        with st.spinner("Thinking..."):
            messages = on_question(st.session_state.messages[-1].content)
            st.session_state.messages.extend(messages)

    # Inject JavaScript to scroll to the bottom
    st.markdown(
        """
        <script>
        function scrollToBottom() {
            var element = document.getElementById("chat-container");
            element.scrollTop = element.scrollHeight;
        }
        window.onload = scrollToBottom;
        </script>
        """,
        unsafe_allow_html=True,
    )

def scrape_article(url: str, max_retries: int = 3) -> Dict[str, str]:
    """Scrape the article content from the given URL with retries using cloudscraper."""
    scraper = cloudscraper.create_scraper()
    for attempt in range(1, max_retries + 1):
        try:
            response = scraper.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find('title').text if soup.find('title') else 'Error'
                content = ' '.join([p.text for p in soup.find_all('p')])
                doi = 'Error'  # Placeholder for DOI, you can add logic to extract DOI if available
                return {'title': title, 'doi': doi, 'content_summary': content}
            else:
                logger.warning(f"Request failed for {url}, status code: {response.status_code}, retrying... ({attempt}/{max_retries})")
        except Exception as e:
            logger.warning(f"Request failed for {url}, retrying... ({attempt}/{max_retries})")
            if attempt == max_retries:
                return {'title': 'Error', 'doi': 'Error', 'content_summary': f'Error scraping after {max_retries} attempts: {url}'}

def extract_content_from_articles(articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Extract and clean content from the articles."""
    cleaned_articles = []
    for article in articles:
        content = article['content_summary']
        # Remove HTML tags and scripts
        soup = BeautifulSoup(content, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        cleaned_articles.append({
            'title': article['title'],
            'doi': article['doi'],
            'content_summary': text
        })
    return cleaned_articles

def st_app(
    title: str = "Turing",
    favicon: str = "🤖",
    image: Optional[str] = None,
    h1: str = "Turing",
    h2: str = "Chat with your docs",
) -> None:
    """Main Streamlit app function."""
    st.set_page_config(
        page_title=title,
        page_icon=favicon,
        layout="wide",
        menu_items={},
    )

    # Initialize visualization agent
    viz_agent = VisualizationAgent()

    # Initialize search agent
    search_agent = SearchAgent()

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        body {
            background-color: white;
            color: black;
            font-family: Arial, sans-serif;
        }
        .stChatMessage {
            border: 1px solid grey;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .stChatMessage .stMarkdown {
            color: gray;
        }
        .stChatInput {
            border: 1px solid grey;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        sidebar_container = st.container()
        st_settings(settings)

    col1, col2 = st.columns([7, 1])
    with col1:
        if h1:
            st.title(h1)
        if h2:
            st.header(h2)
    with col2:
        if image:
            st.image(image)
    st.divider()

    with st.spinner("Loading RAG database, models and chain..."):
        embeddings_model = _get_embeddings_model(
            st.session_state.embeddings_model_name,
            st.session_state.embeddings_model_type,
            device=settings.device,
            trust_remote_code=settings.trust_remote_code,
        )
        llm = _get_llm(
            st.session_state.llm_name,
            st.session_state.llm_type,
            device=settings.device,
            trust_remote_code=settings.trust_remote_code,
            torch_dtype=settings.torch_dtype,
        )
        chain = _get_rag_chain(
            llm,
            st.session_state.relevance_score_fn,
            st.session_state.k,
            st.session_state.search_type,
            st.session_state.score_threshold,
            st.session_state.fetch_k,
            st.session_state.lambda_mult,
            embeddings_model,
        )
        questions_vectorstore = _get_questions_chromadb(embeddings_model)

    viewer = get_or_create_spotlight_viewer()

    def on_question(question: str) -> List[Message]:
        rag_answer = chain.invoke(question)

        messages: List[Message] = []
        sources: List[str] = []
        article_references: List[str] = []

        # Check if visualization is requested and determine the plot type
        plot_type = None
        visualize_requested = any(keyword in question.lower() for keyword in ['visualize', 'plot', 'graph', 'show', 'trend'])

        if visualize_requested:
            st.subheader("Data Visualization")

            # Parse the plot type from the question (if specified)
            if "bar" in question.lower():
                plot_type = "bar"
            elif "scatter" in question.lower():
                plot_type = "scatter"
            else:
                plot_type = "line"  # Default to line plot if no specific type is mentioned

            try:
                viz_agent = VisualizationAgent()  # Ensure the updated class is used
                st.write(f"Debug: Using VisualizationAgent with plot_type={plot_type}")  # Debug logging
                viz_agent.visualize_documents(rag_answer["source_documents"], plot_type=plot_type)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.error("Document structure:")
                st.write(rag_answer["source_documents"][0].__dict__ if rag_answer["source_documents"] else "No documents")
                messages.append(Message("assistant",
                    "I encountered an error while trying to create the visualization. "
                    "Please ensure the documents contain dates and numeric values."))

            for doc in rag_answer["source_documents"]:
                # Access attributes directly instead of using .get()
                sources.append(f"**Content**: {doc.page_content}")
                sources.append(f"**Source**: \"{doc.metadata['source']}\"")

            messages.append(NestedMessage("source", "Sources", sources))
            messages.append(Message("assistant", rag_answer["answer"]))

            return messages

        # Proceed with article extraction if visualization is not requested
        try:
            # Prepare the DataFrame (this is a simplified example)
            df = pd.DataFrame([{
                'timestamp': doc.metadata.get('date', None),
                'value': doc.metadata.get('total_price', None),
                'order_id': doc.metadata.get('order_id', None),
                'source': doc.metadata['source']
            } for doc in rag_answer["source_documents"]])

            # Preprocess the data here directly if preprocess_data is not available
            # Example preprocessing steps:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['value'].fillna(0, inplace=True)
            df = df[df['value'] > 0]

            # Proceed with the search agent analysis
            search_results = search_agent.run(df)
            st.write("Search Agent Results:")
            st.write("Generated Financial Hypotheses:")
            for hypothesis in search_results["hypotheses"]:
                st.write(f"**Hypothesis**: {hypothesis.hypothesis}")
                st.write(f"**Reasoning**: {hypothesis.reasoning}")
                st.write(f"**Potential Impact**: {hypothesis.potential_impact}")
                st.write("---")

            st.write("Found Academic Articles:")
            articles = search_results["articles"]
            cleaned_articles = extract_content_from_articles(articles)
            for article in cleaned_articles:
                st.write(f"**Title**: {article['title']}")
                st.write(f"**DOI**: {article['doi']}")
                st.write(f"**Content Summary**: {article['content_summary']}")
                st.write(f"**Relevance Score**: {article.get('relevance_score', 0.00):.2f}")
                st.write("---")

                # Add article references
                article_references.append(f"**Title**: {article['title']}")
                article_references.append(f"**DOI**: {article['doi']}")
                article_references.append(f"**Content Summary**: {article['content_summary']}")
                article_references.append(f"**Relevance Score**: {article.get('relevance_score', 0.00):.2f}")

            messages.append(Message("assistant", "Here are the financial strategies and academic articles based on the analysis of your data."))

        except Exception as e:
            st.error(f"Error performing financial strategy analysis: {str(e)}")
            messages.append(Message("assistant", "An error occurred while processing the data. Please check the data format."))

        return messages

    def explore() -> None:
        """Display the exploration interface for vectorstores."""
        df = get_docs_questions_df(
            settings.docs_db_directory,
            settings.docs_db_collection,
            settings.questions_db_directory,
            settings.questions_db_collection,
        )
        viewer.show(df, wait=False)

    with sidebar_container:
        if viewer is None:
            st.warning(
                "To explore vectorstores interactively: "
                "`pip install renumics-rag[exploration]` or `pip install renumics-spotlight`",
                icon="⚠️",
            )
        else:
            st.button(
                "Explore",
                help="Explore documents and questions interactively in Renumics Spotlight.",
                on_click=explore,
                type="primary",
                disabled=viewer is None,
            )

    # Wrap the chat messages in a div with an ID for scrolling
    st.markdown(
        """
        <div id="chat-container" style="height: 600px; overflow-y: scroll;">
        """,
        unsafe_allow_html=True,
    )

    st_chat(on_question)

    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True,
    )

app.command()(st_app)

if __name__ == "__main__":
    app()
