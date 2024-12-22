import os
import re
import requests
import pandas as pd
import streamlit as st
import json  # Importing the json module
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tools.internet import google_search, scrape_webpages_with_fallback
import wikipedia  # Use the wikipedia package instead of langchain_wikipedia
import logging

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # You can adjust the level to INFO, WARNING, ERROR based on needs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # You can adjust the level here as well
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Constants
API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment
GOOGLE_SCHOLAR_SEARCH_ENGINE = "https://scholar.google.com/scholar"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Initialize LLM
llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo", api_key=API_KEY)

# Data models
class FinancialHypothesis(BaseModel):
    hypothesis: str = Field(description="A clear and actionable hypothesis focused on improving financial performance")
    reasoning: str = Field(description="The reasoning behind the hypothesis")
    potential_impact: str = Field(description="The potential financial impact of implementing the hypothesis")

class Article(BaseModel):
    title: str = Field(description="The title of the article")
    doi: str = Field(description="The DOI of the article")
    content: str = Field(description="A summary of the article content")
    relevance_score: float = Field(description="A score indicating how relevant the article is to the hypothesis (0-1)")

class StrategyReport(BaseModel):
    executive_summary: str = Field(description="A concise summary of the strategy report")
    financial_strategy: str = Field(description="The financial strategy based on the hypotheses and articles")
    implementation_plan: str = Field(description="A detailed plan for implementing the financial strategy")
    data_integration: str = Field(description="How to integrate the insights from the articles into the company's data")
    potential_challenges: str = Field(description="Potential challenges in implementing the strategy")
    key_metrics: str = Field(description="Key metrics to track the success of the strategy")
    next_steps: str = Field(description="Next steps for the company to follow")

# SearchAgent class
import re

class SearchAgent:
    def __init__(self):
        self.data_cache = {}
        self.hypotheses = []
        self.articles = []

    def _extract_doi(self, content: str) -> str:
        # Use a regular expression to find the DOI in the content
        doi_pattern = r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)"
        match = re.search(doi_pattern, content, re.IGNORECASE)
        if match:
            return match.group(0)  # Return the DOI if found
        return "DOI not found"  # If no DOI is found, return a default message

    def _extract_content_summary(self, content: str) -> str:
        # Extract a brief summary of the content
        # (This can be a placeholder for more complex extraction logic)
        return content[:500]  # Return the first 500 characters of the content as a summary

    def _calculate_relevance(self, query: str, content: str) -> float:
        # Simple relevance calculation based on keyword matches
        keywords = query.split()
        content_words = content.split()
        matched_keywords = sum(1 for word in keywords if word in content_words)
        return matched_keywords / len(keywords) if keywords else 0

    def analyze_data(self, data: pd.DataFrame) -> List[FinancialHypothesis]:
        prompt_template = PromptTemplate(
            input_variables=["data_summary"],
            template="""You are a financial strategist advising a CEO.
            Analyze the following data summary and generate a hypothesis about improving the company's financial performance:
            {data_summary}
            Ensure the hypothesis is clear, actionable, and focused on financial strategies, such as revenue growth, cost reduction, or profitability improvement.
            Provide reasoning for the hypothesis and estimate the potential financial impact.
            Avoid generating hypotheses about data cleaning or data quality issues unless explicitly requested by the user.
            Output in the following JSON format:
            {{
                "hypothesis": "Your hypothesis here",
                "reasoning": "Your reasoning here",
                "potential_impact": "Your potential financial impact here"
            }}
            """,
        )

        data_summary = data.describe().to_string()  # Example: basic statistical summary
        prompt = prompt_template.format(data_summary=data_summary)

        try:
            response = llm.predict(prompt)  # Correct method to invoke LLM in the current version of langchain
            hypothesis_data = response.strip()  # Remove any leading/trailing whitespace
            # Assuming the LLM gives us a string that is JSON formatted
            parsed_response = json.loads(hypothesis_data)  # Parse the hypothesis data into a dict

            # Create a list of FinancialHypothesis objects
            hypotheses = [FinancialHypothesis(**parsed_response)]
            print("Generated Hypotheses:", hypotheses)  # For debugging
        except Exception as e:
            st.error(f"Error generating hypotheses: {e}")
            return []

        self.hypotheses = hypotheses  # Ensure hypotheses is a list of FinancialHypothesis objects
        return self.hypotheses

    def search_articles(self, query: str, num_results: int = 5) -> List[Article]:
        try:
            logger.debug(f"Starting search for query: {query}")  # Log the query being searched
            search_results = google_search(query)  # This now returns a list of dictionaries with URL, title, and snippet
            logger.debug(f"Search results: {search_results}")  # Log the search results
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            st.error(f"Error performing search: {e}")
            return []

        if not search_results:
            logger.warning(f"No search results found for the query: {query}")
            st.warning(f"No search results found for the query: {query}")
            return []

        # Extract URLs from the structured search results
        urls = [result['link'] for result in search_results[:num_results]]
        logger.debug(f"URLs extracted: {urls}")  # Log the extracted URLs

        if not urls:
            logger.warning("No URLs found in the search results.")
            st.warning("No URLs found in the search results.")
            return []

        try:
            # Scraping the webpages with fallback mechanism
            articles_content = scrape_webpages_with_fallback(urls)
            logger.debug(f"Articles content: {articles_content}")  # Log the scraped articles content
            if not articles_content:
                logger.warning("No articles were scraped successfully.")
                st.warning("No articles were scraped successfully.")
                return []

            # Process the scraped content
            articles = []
            for content in articles_content:
                # Ensure we handle missing content correctly
                title = content.get('title', 'Unknown Title')
                doi = content.get('doi', 'Unknown DOI')
                summary = content.get('content_summary', 'No summary available')
                relevance_score = self._calculate_relevance(query, summary)
                articles.append(Article(title=title, doi=doi, content=summary, relevance_score=relevance_score))

            articles.sort(key=lambda x: x.relevance_score, reverse=True)
            return articles[:num_results]  # Only return top articles

        except Exception as e:
            logger.error(f"Error scraping articles: {e}")
            st.error(f"Error scraping articles: {e}")
            return []
    def generate_search_query(self, hypothesis: str) -> str:
        # Generate a more refined search query by extracting key phrases from the hypothesis
        refined_query = " ".join(hypothesis.split(" ")[1:7])  # Take a broader context from the hypothesis
        return refined_query

    import json

    def generate_strategy_report(self, hypotheses: List[FinancialHypothesis], articles: List[Article], data_summary: str) -> StrategyReport:
        if not hypotheses:
            raise ValueError("No financial hypotheses found.")
        if not articles:
            raise ValueError("No academic articles found.")

        # Prepare the hypotheses and articles for the prompt in a structured format
        hypotheses_str = "\n".join([f"Hypothesis: {h.hypothesis}\nReasoning: {h.reasoning}\nPotential Impact: {h.potential_impact}" for h in hypotheses])
        articles_str = "\n".join([f"Title: {a.title}\nDOI: {a.doi}\nContent Summary: {a.content}\nRelevance Score: {a.relevance_score:.2f}" for a in articles])

        prompt_template = PromptTemplate(
            input_variables=["hypotheses", "articles", "data_summary"],
            template="""You are a financial strategist tasked with creating a comprehensive strategy report for a company.
            Based on the following hypotheses, academic articles, and data summary, generate a detailed strategy report:
            Hypotheses:
            {hypotheses}

            Academic Articles:
            {articles}

            Data Summary:
            {data_summary}

            The report should include:
            1. An executive summary of the strategy.
            2. A financial strategy based on the hypotheses and articles.
            3. A detailed implementation plan for the financial strategy.
            4. How to integrate the insights from the articles into the company's data.
            5. Potential challenges in implementing the strategy.
            6. Key metrics to track the success of the strategy.
            7. Next steps for the company to follow.
            Output in the following JSON format:
            {{
                "executive_summary": "Your executive summary here",
                "financial_strategy": "Your financial strategy here",
                "implementation_plan": "Your implementation plan here",
                "data_integration": "Your data integration plan here",
                "potential_challenges": "Your potential challenges here",
                "key_metrics": "Your key metrics here",
                "next_steps": "Your next steps here"
            }}
            """
        )

        prompt = prompt_template.format(hypotheses=hypotheses_str, articles=articles_str, data_summary=data_summary)
        try:
            # Get LLM's response and ensure it's parsed as JSON
            response = llm.predict(prompt).strip()  # Strip excess whitespace
            strategy_report = json.loads(response)

            # Ensure we parse the response correctly
            if isinstance(strategy_report, dict):
                return StrategyReport(**strategy_report)  # Return as a pydantic model
            else:
                raise ValueError("Strategy report response is not in expected JSON format.")

        except Exception as e:
            logger.error(f"Error generating strategy report: {e}")
            st.error(f"Error generating strategy report: {e}")
            return StrategyReport(
                executive_summary="",
                financial_strategy="",
                implementation_plan="",
                data_integration="",
                potential_challenges="",
                key_metrics="",
                next_steps=""
            )
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        st.subheader("Analyzing Data and Generating Financial Hypotheses")
        hypotheses = self.analyze_data(data)
        
        if not hypotheses:
            st.warning("No hypotheses were generated.")
            return {}

        st.write("Generated Financial Hypotheses:")
        for hypothesis in hypotheses:
            st.write(f"**Hypothesis**: {hypothesis.hypothesis}")
            st.write(f"**Reasoning**: {hypothesis.reasoning}")
            st.write(f"**Potential Impact**: {hypothesis.potential_impact}")
            st.write("---")

        st.subheader("Searching for Relevant Academic Articles")
        articles = []
        for hypothesis in hypotheses:
            query = self.generate_search_query(hypothesis.hypothesis)  # Now calling the correct method
            st.write(f"Search query for hypothesis '{hypothesis.hypothesis}': {query}")
            
            # Ensure we get valid articles
            articles += self.search_articles(query)
        
        if not articles:
            st.warning("No articles found.")
            return {}

        st.write("Found Academic Articles:")
        for article in articles:
            st.write(f"**Title**: {article.title}")
            st.write(f"**DOI**: {article.doi}")
            st.write(f"**Content Summary**: {article.content}")
            st.write(f"**Relevance Score**: {article.relevance_score:.2f}")
            st.write("---")

        st.subheader("Generating Strategy Report")
        data_summary = data.describe().to_string()
        strategy_report = self.generate_strategy_report(hypotheses, articles, data_summary)
        
        st.write("Strategy Report:")
        st.write(f"**Executive Summary**: {strategy_report.executive_summary}")
        st.write(f"**Financial Strategy**: {strategy_report.financial_strategy}")
        st.write(f"**Implementation Plan**: {strategy_report.implementation_plan}")
        st.write(f"**Data Integration**: {strategy_report.data_integration}")
        st.write(f"**Potential Challenges**: {strategy_report.potential_challenges}")
        st.write(f"**Key Metrics**: {strategy_report.key_metrics}")
        st.write(f"**Next Steps**: {strategy_report.next_steps}")

        return {
            "hypotheses": hypotheses,
            "articles": self.articles,
            "strategy_report": strategy_report
        }
