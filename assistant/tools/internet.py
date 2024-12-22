import os
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader, FireCrawlLoader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from custom_logger import setup_logger
from load_cfg import FIRECRAWL_API_KEY, CHROMEDRIVER_PATH
import requests
import random
import time

# Set up logger for tracking actions
logger = setup_logger('my_logger')

@tool
def google_search(query: str) -> List[Dict[str, str]]:
    """
    Perform a Google search based on the given query and return the top 5 results as a list of dictionaries.
    """
    try:
        logger.info(f"Performing Google search for query: {query}")
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run headless to avoid opening a window
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        service = Service(CHROMEDRIVER_PATH)

        # Initialize the Chrome WebDriver
        with webdriver.Chrome(options=chrome_options, service=service) as driver:
            url = f"https://www.google.com/search?q={query}"
            logger.debug(f"Accessing URL: {url}")
            driver.get(url)
            html = driver.page_source

        # Parse the HTML to extract search results
        soup = BeautifulSoup(html, 'html.parser')
        search_results = soup.select('.g')
        search_output = []

        # Extract top 5 search results
        for result in search_results[:5]:
            title_element = result.select_one('h3')
            snippet_element = result.select_one('.VwiC3b')
            link_element = result.select_one('a')

            title = title_element.text if title_element else 'No Title'
            snippet = snippet_element.text if snippet_element else 'No Snippet'
            link = link_element['href'] if link_element else 'No Link'

            # Store in a dictionary
            search_output.append({
                'title': title,
                'snippet': snippet,
                'link': link
            })

        logger.info("Google search completed successfully")
        return search_output
    except Exception as e:
        logger.error(f"Error during Google search: {str(e)}")
        return f"Error: {str(e)}"

# Sample user-agent list
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'
]

@tool
def scrape_webpage(url: str) -> str:
    """
    Scrapes the content of a given webpage by sending a GET request to the URL.
    Uses a random user-agent from a predefined list to avoid being blocked.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: The HTML content of the webpage if the request is successful, 
             or an error message if the request fails.
    """
    headers = {
        'User-Agent': random.choice(USER_AGENTS)
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

def scrape_webpages_with_fallback(urls: List[str]) -> List[Dict[str, str]]:
    scraped_results = []
    for url in urls:
        attempt = 1
        while attempt <= 3:
            content = scrape_webpage(url)
            if 'Error' in content or 'Request failed' in content:
                logger.warning(f"Request failed for {url}, retrying... ({attempt}/3)")
                time.sleep(random.uniform(1, 3))  # Random delay between retries
                attempt += 1
            else:
                # Parse the content with BeautifulSoup or process further as needed
                soup = BeautifulSoup(content, 'html.parser')
                # Extract title and summary (or other necessary data)
                title = soup.title.string if soup.title else "No title found"
                scraped_results.append({
                    'title': title,
                    'doi': 'Extracted DOI',  # Example, add actual logic for extracting DOI
                    'content_summary': content[:500]  # Summary of content (first 500 chars)
                })
                break
        else:
            scraped_results.append({
                'title': 'Error',
                'doi': 'Error',
                'content_summary': f"Error scraping after 3 attempts: {url}"
            })
    return scraped_results

@tool
def FireCrawl_scrape_webpages(urls: List[str]) -> str:
    """
    Scrape the provided web pages for detailed information using FireCrawlLoader.
    """
    if not FIRECRAWL_API_KEY:
        logger.error("FireCrawl API key is not set")
        raise ValueError("FireCrawl API key is not set")

    try:
        logger.info(f"Scraping webpages using FireCrawl: {urls}")
        loader = FireCrawlLoader(api_key=FIRECRAWL_API_KEY, url=urls, mode="scrape")
        result = loader.load()  # This could be a string, but you're expecting a dict
        logger.info("FireCrawl scraping completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during FireCrawl scraping: {str(e)}")
        return f"Error: {str(e)}"

def scrape_content(url: str) -> str:
    """
    Scrape content from a single webpage with retry mechanism.
    """
    retries = 3
    for attempt in range(retries):
        try:
            logger.info(f"Scraping content from URL: {url}, attempt {attempt + 1}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Ensure the request was successful
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()  # Extract all text from the webpage
            logger.info(f"Successfully scraped content from {url}")
            return text
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:  # If not the last attempt
                logger.warning(f"Request failed for {url}, retrying... ({attempt + 1}/{retries})")
                time.sleep(2)  # Wait before retrying
            else:
                logger.error(f"Request failed for {url} after {retries} attempts: {str(e)}")
                return f"Error: {str(e)}"
