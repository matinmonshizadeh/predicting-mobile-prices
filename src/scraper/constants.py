from pathlib import Path

# Divar mobile-phone listings for Tehran.
BASE_URL = "https://divar.ir/s/tehran/mobile-phones"

# Where scraped rows are appended.
OUTPUT_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "item_details.csv"

# Detail-row titles exactly as Divar renders them. These become the CSV columns
# and src/features.py maps them to English names.
CSV_HEADERS = [
    "برند و مدل",  # brand and model
    "وضعیت",  # condition
    "تعداد سیم‌کارت",  # SIM count
    "اصالت برند",  # brand origin (genuine / not genuine)
    "حافظهٔ داخلی",  # internal storage
    "مقدار رم",  # RAM
    "قیمت",  # price
    "رنگ",  # colour
]

MAX_ITEMS = 3500  # stop after this many listings
SCROLL_DELAY = 3  # seconds between page loads, to stay polite
PAGE_TIMEOUT = 10  # seconds to wait for elements
MAX_EMPTY_ROUNDS = 3  # give up after this many "load more" clicks with no new links

# CSS selectors. Divar's listing page uses hashed class names that change with
# every deploy (the original selectors from July 2024 no longer exist), so the
# listing selectors match on stable prefixes. Last checked: September 2026.
ITEM_LINK_SELECTOR = "a.kt-post-card__action[href^='/v/']"
LOAD_MORE_BUTTON_SELECTOR = "button[class*='post-list__load-more-btn']"
DETAIL_SECTION_SELECTOR = "div.post-page__section--padded"
DETAIL_ROW_SELECTOR = "div.kt-base-row"
ROW_TITLE_SELECTOR = "p.kt-base-row__title"
ROW_VALUE_SELECTOR = "p.kt-unexpandable-row__value"
ROW_LINK_VALUE_SELECTOR = "a.kt-unexpandable-row__action"
BRAND_MODEL_SELECTOR = "div.kt-base-row__end a"
