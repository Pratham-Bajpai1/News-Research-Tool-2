from typing import List, Optional
from langchain.schema import Document
import aiohttp
from bs4 import BeautifulSoup

async def fetch_url_content(url: str) -> Optional[str]:
    """Fetch URL content with error handling"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as response:
                response.raise_for_status()
                return await response.text()
    except Exception:
        return None

def clean_html_content(html: str) -> str:
    """Clean HTML content"""
    soup = BeautifulSoup(html, 'html.parser')
    for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
        element.decompose()
    return ' '.join(soup.get_text(separator=' ', strip=True).split())
