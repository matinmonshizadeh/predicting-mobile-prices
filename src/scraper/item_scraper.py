import time

from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from constants import (
    BRAND_MODEL_SELECTOR,
    DETAIL_ROW_SELECTOR,
    DETAIL_SECTION_SELECTOR,
    ITEM_LINK_SELECTOR,
    LOAD_MORE_BUTTON_SELECTOR,
    PAGE_TIMEOUT,
    ROW_LINK_VALUE_SELECTOR,
    ROW_TITLE_SELECTOR,
    ROW_VALUE_SELECTOR,
    SCROLL_DELAY,
)


def retrieve_item_links(driver) -> list[str]:
    """Return every listing URL currently on the page, in page order, without
    duplicates. The caller decides which ones are new."""
    try:
        WebDriverWait(driver, PAGE_TIMEOUT).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ITEM_LINK_SELECTOR))
        )
    except TimeoutException:
        print("Timeout waiting for item links to load")
        return []

    for _attempt in range(3):
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, ITEM_LINK_SELECTOR)
            hrefs = [el.get_attribute("href") for el in elements]
            break
        except StaleElementReferenceException:
            # The list re-rendered while we were reading it; read it again.
            time.sleep(1)
    else:
        return []
    return list(dict.fromkeys(h for h in hrefs if h))


def scroll_and_load_more_items(driver) -> None:
    """Scroll to the bottom and click "load more" if it is showing."""
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    try:
        button = driver.find_element(By.CSS_SELECTOR, LOAD_MORE_BUTTON_SELECTOR)
        if button.is_displayed():
            button.click()
            print("Clicked 'load more'")
    except NoSuchElementException:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"Could not click 'load more': {exc}")
    time.sleep(SCROLL_DELAY)


def scrape_item_details(driver, link: str) -> dict[str, str]:
    """Open a listing in a new tab, read its detail rows, close the tab."""
    main_tab = driver.current_window_handle
    before = set(driver.window_handles)
    driver.execute_script("window.open(arguments[0], '_blank');", link)
    new_tab = (set(driver.window_handles) - before).pop()
    driver.switch_to.window(new_tab)

    item: dict[str, str] = {}
    try:
        section = WebDriverWait(driver, PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, DETAIL_SECTION_SELECTOR))
        )
        try:
            item["برند و مدل"] = section.find_element(By.CSS_SELECTOR, BRAND_MODEL_SELECTOR).text.strip()
        except NoSuchElementException:
            pass

        for row in section.find_elements(By.CSS_SELECTOR, DETAIL_ROW_SELECTOR):
            try:
                title = row.find_element(By.CSS_SELECTOR, ROW_TITLE_SELECTOR).text.strip()
            except NoSuchElementException:
                continue
            try:
                value = row.find_element(By.CSS_SELECTOR, ROW_VALUE_SELECTOR).text.strip()
            except NoSuchElementException:
                try:
                    value = row.find_element(By.CSS_SELECTOR, ROW_LINK_VALUE_SELECTOR).text.strip()
                except NoSuchElementException:
                    continue
            item[title] = value
    except TimeoutException:
        print(f"Timeout waiting for details: {link}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error reading {link}: {exc}")
    finally:
        driver.close()
        driver.switch_to.window(main_tab)
    return item
