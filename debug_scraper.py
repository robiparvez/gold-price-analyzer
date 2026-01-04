"""
Debug script for testing gold price scraping
"""

import requests
from bs4 import BeautifulSoup


def debug_scraping():
    """Debug the scraping process step by step."""
    url = "https://www.bajus.org/gold-price"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print("Fetching webpage...")
    response = requests.get(url, headers=headers, timeout=30)
    print(f"Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Find all tables
    tables = soup.find_all("table")
    print(f"Found {len(tables)} tables")

    # Print the raw HTML structure for debugging
    for i, table in enumerate(tables):
        print(f"\n--- Table {i + 1} ---")
        print(table.prettify()[:500])  # First 500 characters

        rows = table.find_all("tr")
        print(f"Table {i + 1} has {len(rows)} rows")

        for j, row in enumerate(rows[:3]):  # First 3 rows
            cells = row.find_all(["td", "th"])
            print(f"Row {j + 1}: {len(cells)} cells")
            for k, cell in enumerate(cells):
                text = cell.get_text(strip=True)
                print(f"  Cell {k + 1}: '{text}'")


if __name__ == "__main__":
    debug_scraping()
