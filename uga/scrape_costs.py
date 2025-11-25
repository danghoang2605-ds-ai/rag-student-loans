from bs4 import BeautifulSoup
import json
import requests

# URL of the UGA Cost of Attendance page
url = "https://osfa.uga.edu/costs/"

# Fetch and parse the HTML using html5lib for robustness
html = requests.get(url).text
soup = BeautifulSoup(html, "html5lib")

# Scope to the main content area to avoid irrelevant tables
content = soup.select_one("div.container div.section.group")
tables = content.find_all("table") if content else []

data = []

for table in tables:
    # Try to get the section heading (h2 or h3) immediately before this table
    heading_tag = table.find_previous(["h2", "h3"])
    heading = heading_tag.get_text(strip=True) if heading_tag else "Unknown Section"

    # Extract headers (if present)
    headers = [th.get_text(strip=True) for th in table.find_all("th")]

    # Extract all table rows
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)

    data.append({
        "heading": heading,
        "headers": headers,
        "rows": rows
    })

# Write structured data to JSON
with open("uga_cost_tables.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Extracted {len(data)} tables and saved to uga_cost_tables.json")
