"""Best-effort keep-alive for a Streamlit Community Cloud app.

Why a real browser: a Streamlit app is a JavaScript single-page app that talks to the
server over a WebSocket. A bare ``curl``/``requests`` GET only fetches the static HTML
shell and may not count as a visitor. This script opens the page in headless Chromium,
clicks the "get this app back up" button if the app is already asleep, then waits until
the app has rendered.

Usage:
    python scripts/keep_alive.py https://YOUR-APP.streamlit.app

Exit code 0 = app rendered, 1 = it did not (the GitHub Actions run will show red).

This is *best effort*. It relies on Streamlit Community Cloud's current inactivity policy
and page markup, both of which Streamlit may change at any time.
"""
from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import Page, sync_playwright

WAKE_BUTTON_TEXT = "get this app back up"  # matches 'Yes, get this app back up!'
APP_SELECTOR = '[data-testid="stAppViewContainer"]'
HOSTED_IFRAME = 'iframe[title="streamlitApp"]'  # Community Cloud embeds the app in an iframe


def wake_if_asleep(page: Page) -> bool:
    """Click the wake-up button when present. Returns True if it was clicked."""
    button = page.get_by_role("button", name=WAKE_BUTTON_TEXT, exact=False)
    try:
        button.first.wait_for(state="visible", timeout=8_000)
    except Exception:
        return False
    button.first.click()
    print("App was asleep: clicked the wake-up button.")
    return True


def app_rendered(page: Page) -> bool:
    """True once the Streamlit app container is visible (inside the iframe when hosted)."""
    if page.locator(APP_SELECTOR).count() > 0:
        return True
    return page.frame_locator(HOSTED_IFRAME).locator(APP_SELECTOR).count() > 0


def main(url: str, timeout_s: int = 240) -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)  # CHROMIUM_PATH: optional local override
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        wake_if_asleep(page)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if app_rendered(page):
                time.sleep(3)  # stay connected briefly so the visit is registered
                print(f"OK: app rendered at {url}")
                browser.close()
                return 0
            time.sleep(3)
        print(f"FAILED: app did not render within {timeout_s}s at {url}", file=sys.stderr)
        browser.close()
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
