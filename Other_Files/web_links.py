import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Set, Dict, List, Optional
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@dataclass
class CrawlStats:
    total_pages_crawled: int = 0
    total_links_found: int = 0
    internal_links: int = 0
    external_links: int = 0
    broken_links: int = 0
    pages_with_errors: int = 0
    link_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.link_types is None:
            self.link_types = defaultdict(int)

class RobustWebCrawler:
    def __init__(self, 
                 max_pages: int = 100,
                 max_depth: int = 3,
                 delay: float = 1.0,
                 respect_robots: bool = True,
                 user_agent: str = "RobustCrawler/1.0",
                 timeout: int = 10,
                 max_retries: int = 3):
        
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.respect_robots = respect_robots
        self.timeout = timeout
        self.user_agent = user_agent
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Crawl state
        self.visited_urls: Set[str] = set()
        self.url_queue: deque = deque()
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.stats = CrawlStats()
        
    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and query params (optional)"""
        parsed = urlparse(url)
        # Remove fragment
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,  # Keep query params - remove this line to ignore them
            ''  # Remove fragment
        ))
        return normalized.rstrip('/')
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and crawlable"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Skip non-HTTP protocols
            if parsed.scheme not in ['http', 'https']:
                return False
                
            # Skip common file extensions
            skip_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', 
                             '.zip', '.rar', '.exe', '.doc', '.docx', '.xls', '.xlsx'}
            if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
                return False
                
            return True
        except Exception:
            return False
    
    def can_fetch(self, url: str) -> bool:
        """Check robots.txt permissions"""
        if not self.respect_robots:
            return True
            
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            if base_url not in self.robots_cache:
                robots_url = urljoin(base_url, '/robots.txt')
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self.robots_cache[base_url] = rp
                except Exception:
                    # If can't read robots.txt, assume allowed
                    return True
            
            return self.robots_cache[base_url].can_fetch(self.user_agent, url)
        except Exception:
            return True
    
    def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract all links from HTML content"""
        links = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all anchor tags with href
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href:
                    # Resolve relative URLs
                    full_url = urljoin(base_url, href)
                    normalized_url = self.normalize_url(full_url)
                    
                    if self.is_valid_url(normalized_url):
                        links.append(normalized_url)
                        
                        # Categorize link types
                        if urlparse(normalized_url).netloc == urlparse(base_url).netloc:
                            self.stats.internal_links += 1
                        else:
                            self.stats.external_links += 1
                            
                        # Track link types
                        link_text = link.get_text(strip=True)[:50]  # First 50 chars
                        if link_text:
                            self.stats.link_types[f"Text: {link_text}"] += 1
                        else:
                            self.stats.link_types["No text"] += 1
            
            # Also check for other link sources
            for tag in soup.find_all(['link', 'area']):
                href = tag.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    normalized_url = self.normalize_url(full_url)
                    if self.is_valid_url(normalized_url):
                        links.append(normalized_url)
                        
        except Exception as e:
            self.logger.error(f"Error extracting links from {base_url}: {e}")
        
        return links
    
    def crawl_page(self, url: str) -> Optional[List[str]]:
        """Crawl a single page and return found links"""
        try:
            if not self.can_fetch(url):
                self.logger.info(f"Robots.txt disallows crawling: {url}")
                return None
            
            self.logger.info(f"Crawling: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                self.logger.info(f"Skipping non-HTML content: {url}")
                return None
            
            links = self.extract_links(response.text, url)
            self.stats.total_links_found += len(links)
            self.stats.total_pages_crawled += 1
            
            return links
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for {url}: {e}")
            self.stats.pages_with_errors += 1
            self.stats.broken_links += 1
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error crawling {url}: {e}")
            self.stats.pages_with_errors += 1
            return None
    
    def crawl_website(self, start_url: str, same_domain_only: bool = True) -> CrawlStats:
        """Main crawling method"""
        start_url = self.normalize_url(start_url)
        start_domain = urlparse(start_url).netloc
        
        # Initialize queue with start URL and depth 0
        self.url_queue.append((start_url, 0))
        
        self.logger.info(f"Starting crawl of {start_url}")
        self.logger.info(f"Max pages: {self.max_pages}, Max depth: {self.max_depth}")
        
        while self.url_queue and len(self.visited_urls) < self.max_pages:
            current_url, depth = self.url_queue.popleft()
            
            # Skip if already visited or too deep
            if current_url in self.visited_urls or depth > self.max_depth:
                continue
            
            # Skip if different domain (when same_domain_only is True)
            if same_domain_only and urlparse(current_url).netloc != start_domain:
                continue
            
            self.visited_urls.add(current_url)
            
            # Crawl the page
            found_links = self.crawl_page(current_url)
            
            if found_links:
                # Add new links to queue for next depth level
                for link in found_links:
                    if link not in self.visited_urls:
                        self.url_queue.append((link, depth + 1))
            
            # Respectful delay
            if self.delay > 0:
                time.sleep(self.delay)
        
        self.logger.info(f"Crawl completed. Visited {len(self.visited_urls)} pages")
        return self.stats
    
    def print_report(self):
        """Print detailed crawling report"""
        print("\n" + "="*60)
        print("CRAWLING REPORT")
        print("="*60)
        print(f"Total pages crawled: {self.stats.total_pages_crawled}")
        print(f"Total links found: {self.stats.total_links_found}")
        print(f"Internal links: {self.stats.internal_links}")
        print(f"External links: {self.stats.external_links}")
        print(f"Pages with errors: {self.stats.pages_with_errors}")
        print(f"Broken/inaccessible links: {self.stats.broken_links}")
        
        if self.stats.total_pages_crawled > 0:
            avg_links = self.stats.total_links_found / self.stats.total_pages_crawled
            print(f"Average links per page: {avg_links:.2f}")
        
        print("\nTop link types:")
        for link_type, count in sorted(self.stats.link_types.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {link_type}: {count}")

# Usage example
if __name__ == "__main__":
    # Configure the crawler
    crawler = RobustWebCrawler(
        max_pages=50,        # Maximum pages to crawl
        max_depth=2,         # Maximum depth from start URL
        delay=1.0,           # Delay between requests (seconds)
        respect_robots=True, # Respect robots.txt
        timeout=10,          # Request timeout
        max_retries=3        # Retry failed requests
    )
    
    # Start crawling
    start_url = "https://maryvillecollege.smartcatalogiq.com/en/current/academic-catalog/"  # Replace with target URL
    stats = crawler.crawl_website(start_url, same_domain_only=True)
    
    # Print detailed report
    crawler.print_report()
    
    # Export visited URLs (optional)
    print(f"\nVisited URLs ({len(crawler.visited_urls)}):")
    for i, url in enumerate(sorted(crawler.visited_urls), 1):
        print(f"{i:3d}. {url}")