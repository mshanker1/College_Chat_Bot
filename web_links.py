import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

def count_links(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=True)
        return len(links)
    except Exception as e:
        print(f"Error: {e}")
        return 0

# For single page
link_count = count_links("https://maryvillecollege.smartcatalogiq.com/en/current/academic-catalog/")
print(f"Links found: {link_count}")