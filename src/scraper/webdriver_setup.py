from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def initialize_webdriver(headless: bool = False) -> webdriver.Chrome:
    """Start Chrome. Selenium 4.6+ downloads a matching chromedriver itself
    (Selenium Manager), so no driver binary is shipped with the repo."""
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--incognito")
    if headless:
        options.add_argument("--headless=new")
    return webdriver.Chrome(options=options)
