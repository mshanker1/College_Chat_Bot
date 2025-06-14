#!/usr/bin/env python3
"""
Maryville College Website Crawler with PDF Generator
Crawls the Maryville College website and generates a comprehensive PDF report.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import re
from collections import defaultdict
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import logging
from datetime import datetime
import os

class MaryvilleCollegeCrawler:
    def __init__(self, base_url="https://www.maryvillecollege.edu/"):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.visited_urls = set()
        self.crawled_data = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research Bot) MaryvilleCollegeCrawler/1.0'
        })
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.delay = 1  # seconds between requests
        
    def check_robots_txt(self):
        """Check robots.txt compliance"""
        try:
            rp = RobotFileParser()
            rp.set_url(urljoin(self.base_url, '/robots.txt'))
            rp.read()
            return rp.can_fetch('*', self.base_url)
        except Exception as e:
            self.logger.warning(f"Could not check robots.txt: {e}")
            return True
    
    def is_valid_url(self, url):
        """Check if URL is valid for crawling"""
        parsed = urlparse(url)
        
        # Only crawl URLs from the same domain
        if parsed.netloc != self.domain:
            return False
            
        # Skip certain file types
        skip_extensions = ['.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png', '.gif', 
                          '.zip', '.mp4', '.mp3', '.css', '.js']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        # Skip common non-content URLs
        skip_patterns = ['/wp-admin/', '/wp-content/', '/admin/', '/login/', 
                        '/search/', '?', '#']
        if any(pattern in url.lower() for pattern in skip_patterns):
            return False
            
        return True
    
    def extract_page_data(self, url, soup):
        """Extract relevant data from a page"""
        data = {
            'url': url,
            'title': '',
            'meta_description': '',
            'headings': [],
            'content': '',
            'links': [],
            'images': [],
            'contact_info': []
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            data['title'] = title_tag.get_text().strip()
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            data['meta_description'] = meta_desc.get('content', '').strip()
        
        # Extract headings
        for i in range(1, 7):
            headings = soup.find_all(f'h{i}')
            for heading in headings:
                text = heading.get_text().strip()
                if text:
                    data['headings'].append({
                        'level': i,
                        'text': text
                    })
        
        # Extract main content (paragraphs)
        paragraphs = soup.find_all('p')
        content_parts = []
        for p in paragraphs:
            text = p.get_text().strip()
            if text and len(text) > 20:  # Filter out very short text
                content_parts.append(text)
        data['content'] = '\n\n'.join(content_parts[:5])  # Limit content
        
        # Extract links
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            text = link.get_text().strip()
            if href and text:
                full_url = urljoin(url, href)
                data['links'].append({
                    'url': full_url,
                    'text': text
                })
        
        # Extract contact information
        contact_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        
        page_text = soup.get_text()
        for pattern in contact_patterns:
            matches = re.findall(pattern, page_text)
            data['contact_info'].extend(matches)
        
        return data
    
    def crawl_page(self, url):
        """Crawl a single page"""
        if url in self.visited_urls or not self.is_valid_url(url):
            return []
        
        try:
            self.logger.info(f"Crawling: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            self.visited_urls.add(url)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            page_data = self.extract_page_data(url, soup)
            self.crawled_data.append(page_data)
            
            # Find new URLs to crawl
            new_urls = []
            for link_data in page_data['links']:
                link_url = link_data['url']
                if (link_url not in self.visited_urls and 
                    self.is_valid_url(link_url) and 
                    len(new_urls) < 5):  # Limit new URLs per page
                    new_urls.append(link_url)
            
            # Rate limiting
            time.sleep(self.delay)
            
            return new_urls
            
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {e}")
            return []
    
    def crawl_website(self, max_pages=20):
        """Crawl the website starting from the base URL"""
        if not self.check_robots_txt():
            self.logger.warning("Robots.txt may restrict crawling")
        
        self.logger.info("Starting website crawl...")
        
        urls_to_visit = [self.base_url]
        pages_crawled = 0
        
        while urls_to_visit and pages_crawled < max_pages:
            current_url = urls_to_visit.pop(0)
            new_urls = self.crawl_page(current_url)
            
            # Add new URLs to visit queue
            for url in new_urls:
                if url not in urls_to_visit and len(urls_to_visit) < 50:
                    urls_to_visit.append(url)
            
            pages_crawled += 1
            
            if pages_crawled % 5 == 0:
                self.logger.info(f"Crawled {pages_crawled} pages so far...")
        
        self.logger.info(f"Crawling completed. Total pages: {len(self.crawled_data)}")
    
    def generate_pdf_report(self, filename="maryville_college_crawl_report.pdf"):
        """Generate a comprehensive PDF report"""
        doc = SimpleDocTemplate(filename, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkgreen
        )
        
        story = []
        
        # Title page
        story.append(Paragraph("Maryville College Website Analysis Report", title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
        story.append(Paragraph(f"Total pages crawled: {len(self.crawled_data)}", styles['Normal']))
        story.append(Paragraph(f"Base URL: {self.base_url}", styles['Normal']))
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", title_style))
        
        # Collect statistics
        total_pages = len(self.crawled_data)
        total_headings = sum(len(page['headings']) for page in self.crawled_data)
        total_links = sum(len(page['links']) for page in self.crawled_data)
        contact_info = set()
        for page in self.crawled_data:
            contact_info.update(page['contact_info'])
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Pages Analyzed', str(total_pages)],
            ['Total Headings Found', str(total_headings)],
            ['Total Links Found', str(total_links)],
            ['Unique Contact Info Items', str(len(contact_info))],
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(PageBreak())
        
        # Contact Information
        if contact_info:
            story.append(Paragraph("Contact Information Found", heading_style))
            for info in sorted(contact_info):
                story.append(Paragraph(f"• {info}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Page Details
        story.append(Paragraph("Page Analysis Details", title_style))
        
        for i, page in enumerate(self.crawled_data[:10]):  # Limit to first 10 pages
            story.append(Paragraph(f"Page {i+1}: {page['title']}", heading_style))
            story.append(Paragraph(f"<b>URL:</b> {page['url']}", styles['Normal']))
            
            if page['meta_description']:
                story.append(Paragraph(f"<b>Description:</b> {page['meta_description']}", styles['Normal']))
            
            # Headings
            if page['headings']:
                story.append(Paragraph("<b>Main Headings:</b>", styles['Normal']))
                for heading in page['headings'][:5]:  # Limit headings
                    story.append(Paragraph(f"{'  ' * (heading['level']-1)}• {heading['text']}", styles['Normal']))
            
            # Content preview
            if page['content']:
                content_preview = page['content'][:300] + "..." if len(page['content']) > 300 else page['content']
                story.append(Paragraph(f"<b>Content Preview:</b> {content_preview}", styles['Normal']))
            
            story.append(Spacer(1, 12))
            
            if i < len(self.crawled_data) - 1:  # Don't add page break after last page
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        self.logger.info(f"PDF report generated: {filename}")
        return filename

def main():
    """Main function to run the crawler"""
    print("Maryville College Website Crawler")
    print("=" * 50)
    
    # Initialize crawler
    crawler = MaryvilleCollegeCrawler()
    
    try:
        # Crawl the website
        crawler.crawl_website(max_pages=15)  # Limit to 15 pages for demo
        
        # Generate PDF report
        pdf_filename = crawler.generate_pdf_report()
        
        print(f"\nCrawling completed successfully!")
        print(f"Pages crawled: {len(crawler.crawled_data)}")
        print(f"PDF report saved as: {pdf_filename}")
        
    except KeyboardInterrupt:
        print("\nCrawling interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()