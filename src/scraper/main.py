"""Scrape used-phone listings from divar.ir into data/raw/item_details.csv.

    python src/scraper/main.py

Divar changes its page markup regularly, so the selectors in constants.py may
need updating before this runs. Keep SCROLL_DELAY at a polite value.
"""

import time

from constants import BASE_URL, CSV_HEADERS, MAX_EMPTY_ROUNDS, MAX_ITEMS, OUTPUT_CSV, SCROLL_DELAY
from csv_handler import CSVWriter
from item_scraper import retrieve_item_links, scrape_item_details, scroll_and_load_more_items
from webdriver_setup import initialize_webdriver


def main() -> None:
    driver = initialize_webdriver()
    writer = CSVWriter(OUTPUT_CSV, CSV_HEADERS)
    seen: set[str] = set()
    saved = 0
    empty_rounds = 0

    try:
        driver.get(BASE_URL)
        writer.open()

        while saved < MAX_ITEMS and empty_rounds < MAX_EMPTY_ROUNDS:
            new_links = [link for link in retrieve_item_links(driver) if link not in seen]
            if not new_links:
                empty_rounds += 1
                scroll_and_load_more_items(driver)
                continue
            empty_rounds = 0

            for link in new_links:
                seen.add(link)
                item = scrape_item_details(driver, link)
                row = [item.get(header, "") for header in CSV_HEADERS]
                writer.write_row(row)
                saved += 1
                print(f"[{saved}] {row}")
                time.sleep(SCROLL_DELAY)
                if saved >= MAX_ITEMS:
                    break

            scroll_and_load_more_items(driver)
    finally:
        writer.close()
        driver.quit()
        print(f"Done. {saved} listings written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
