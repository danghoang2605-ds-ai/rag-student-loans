import os
import json
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

BASE_URL = "https://osfa.uga.edu/resources/"
OUTPUT_FILE = "../data/uga_resources_all.json"

#skip unwanted sections
SKIP_PREFIXES = [
    "https://osfa.uga.edu/resources/faqs"
]

def allowed(u: str) -> bool:
    """Ensure URL stays within the osfa.uga.edu domain."""
    try:
        p = urlparse(u)
        return p.netloc == "osfa.uga.edu"
    except Exception:
        return False


def scrape_page(url: str, page):
    print(f"  [SCRAPE] {url}")
    docs = []
    try:
        page.goto(url, timeout=180000)
        page.wait_for_selector("main#main, div.entry, article, div.content", timeout=20000)
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        #remove nav/footer/irrelevant sections
        for tag in soup.select("nav, footer, header, aside, .breadcrumbs, .sidebar, script, style"):
            tag.decompose()

        # find main content 
        content = soup.select_one(
            "div.entry, div.content, main#main, article, div.post, div.tables_content, div.section_group"
        ) or soup

        page_title = soup.title.get_text(strip=True) if soup.title else "UGA OSFA Resource"
        chunks = []
        current_chunk = []
        chunk_contains_table = False
        header_text = page_title


        def flush_chunk(title_hint=None):
            nonlocal chunk_contains_table
            text = " ".join(current_chunk).strip()
            min_length = 0 if chunk_contains_table else 200
            if len(text) > min_length:
                chunks.append({
                    "section_title": title_hint or page_title,
                    "text": text
                })
            current_chunk.clear()
            chunk_contains_table = False

        # walk through meaningful tags
        for elem in content.find_all(["h1", "h2", "h3", "p", "li", "table"]):
            if elem.name in ["h1", "h2", "h3"]:
                # Flush previous section before new header
                if current_chunk:
                    flush_chunk(title_hint=header_text)
                header_text = elem.get_text(" ", strip=True)
                continue

            elif elem.name == "table":
                rows = []
                for tr in elem.find_all("tr"):
                    cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    table_text = "[TABLE START]\n" + "\n".join(rows) + "\n[TABLE END]"
                    current_chunk.append(table_text)
                    chunk_contains_table = True

            else:
                txt = elem.get_text(" ", strip=True)
                if txt and txt.lower() != "nbsp":
                    current_chunk.append(txt)

            # chunk by length
            if sum(len(s) for s in current_chunk) > 600:
                flush_chunk(title_hint=header_text)

        if current_chunk:
            flush_chunk(title_hint=header_text)

        def build_question_title(section_title: str) -> str:
            """
            Combine the page title with short/ambiguous section headers so the downstream
            question text is easier to understand (e.g., avoid single-letter headings).
            """
            section_title = (section_title or "").strip()
            if not section_title:
                return page_title

            normalized_page = page_title.strip()
            if not normalized_page:
                return section_title

            #skip prefixing if the section already includes the page title
            if normalized_page.lower() in section_title.lower():
                return section_title

            #attach the page title when the section name is short or generic.
            if len(section_title) <= 3 or len(section_title.split()) <= 2:
                return f"{normalized_page} - {section_title}"

            return section_title

        # Convert chunks to docs
        for ch in chunks:
            doc = {
                "question": build_question_title(ch["section_title"]),
                "answer": ch["text"],
                "url": url
            }
            print(doc)
            docs.append(doc)

        if docs:
            print(f"    [+] Extracted {len(docs)} section(s)")
        else:
            print("    [!] No text extracted")

    except Exception as e:
        print(f"    [!] scrape_page failed: {e}")

    return docs





# Comprehensive set of selectors for link discovery
LINK_SELECTORS = [
    # Standard resource boxes
    "div.portal_item div.box_hover a",
    # 'Images portal' layout
    "div.images_portal div.portal div.portal_item div.icon_wrap div.box_hover a",
    # Title card links
    "div.portal_item h4.post_title a",
    # Inline or content-area resource links
    "main#main a[href]",
    "div.entry a[href]",
    "div.content a[href]",
    # Announcements, articles, or post-style pages
    "article a[href]",
    "div.post a[href]",
]


def crawl_and_scrape(browser):
    """Breadth-first crawl: visit each page once, scrape all content."""
    visited = set()
    all_docs = []
    queue = deque([BASE_URL])

    while queue:
        url = queue.popleft()
        norm = url.split("#")[0]
        if norm in visited:
            continue

        if any(norm.startswith(prefix) for prefix in SKIP_PREFIXES):
            print(f"Skipping: {url}")
            visited.add(norm)
            continue

        visited.add(norm)
        print(f"\nVisiting: {url}")

        page = browser.new_page()
        page.set_default_timeout(180000)
        page.set_default_navigation_timeout(180000)

        # retry navigation if too slow
        success = False
        for attempt in range(3):
            try:
                page.goto(url, timeout=180000)
                success = True
                break
            except Exception as e:
                print(f"  [Retry {attempt+1}] Failed to load {url}: {e}")
                time.sleep(5)
        if not success:
            page.close()
            continue

        # scrape this page
        docs = scrape_page(url, page)
        if docs:
            all_docs.extend(docs)

        # find subpage links to crawl
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        found_links = []
        for selector in LINK_SELECTORS:
            for a in soup.select(selector):
                href = a.get("href")
                if not href or href.startswith("#"):
                    continue
                full = urljoin(url, href)
                if allowed(full) and "/resources/" in full:
                    found_links.append(full)

        found_links = sorted(set(found_links))
        if found_links:
            print(f"  + {len(found_links)} link(s) found")
            for link in found_links:
                if link not in visited:
                    queue.append(link)

        page.close()
        time.sleep(4)

    return all_docs, visited


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        all_docs, visited = crawl_and_scrape(browser)
        browser.close()

    all_docs = [d for d in all_docs if len(d.get("answer","")) >= 120]
    seen = set()
    unique = []
    for d in all_docs:
        key = (d.get("url",""), d.get("question","").strip(), d.get("answer","").strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    all_docs = unique



    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_docs, f, indent=2)

    print(f"\nDone. Visited {len(visited)} pages, scraped {len(all_docs)} sections.")
    print(f"Saved to {OUTPUT_FILE}")
