"""Best-effort keep-alive for a Streamlit Community Cloud app.

Why a real browser: a Streamlit app is a JavaScript single-page app that talks to the
server over a WebSocket. A bare curl/requests GET only fetches the static HTML shell and
does not count as a visitor. This script opens the page in headless Chrome, clicks the
"get this app back up" button if the app is asleep, then waits until the app has really
rendered (inside the Community Cloud iframe) and stays connected briefly.

Usage:
    python scripts/keep_alive.py https://YOUR-APP.streamlit.app

Exit code 0 = app rendered, 1 = it did not. On failure a screenshot is saved to
keep_alive_debug.png (uploaded as a workflow artifact) so you can see what the page showed.
"""
from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

WAKE_BUTTON_TEXT = "get this app back up"        # matches 'Yes, get this app back up!'
APP_SELECTOR = '[data-testid="stAppViewContainer"]'
HOSTED_IFRAME = 'iframe[title="streamlitApp"]'   # Community Cloud embeds the app in an iframe
DEBUG_PNG = "keep_alive_debug.png"
ATTEMPTS = 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def launch(p):
    """Prefer the runner's preinstalled Chrome; fall back to Playwright's bundled Chromium."""
    override = os.environ.get("CHROMIUM_PATH")
    if override:
        return p.chromium.launch(executable_path=override)
    try:
        return p.chromium.launch(channel="chrome")
    except PlaywrightError as exc:
        log(f"System Chrome unavailable ({exc.__class__.__name__}); trying bundled Chromium.")
        return p.chromium.launch()


def wake_if_asleep(page: Page) -> bool:
    button = page.get_by_role("button", name=WAKE_BUTTON_TEXT, exact=False)
    try:
        button.first.wait_for(state="visible", timeout=10_000)
    except PlaywrightError:
        return False
    button.first.click()
    log("App was asleep: clicked the wake-up button.")
    return True


def app_rendered(page: Page) -> bool:
    """True once the Streamlit app container is visible (top level, or inside the iframe)."""
    try:
        if page.locator(APP_SELECTOR).first.is_visible():
            return True
        return page.frame_locator(HOSTED_IFRAME).locator(APP_SELECTOR).first.is_visible()
    except PlaywrightError:
        return False


def attempt(url: str, timeout_s: int) -> bool:
    with sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page()
        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            log(f"Loaded {page.url} (HTTP {resp.status if resp else '?'}), title={page.title()!r}")
            wake_if_asleep(page)
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                if app_rendered(page):
                    time.sleep(10)  # stay connected so the websocket visit is registered
                    log(f"OK: app rendered at {url}")
                    return True
                time.sleep(3)
            log(f"App did not render within {timeout_s}s.")
            return False
        except PlaywrightError as exc:
            log(f"Browser error: {exc}")
            return False
        finally:
            try:
                page.screenshot(path=DEBUG_PNG, full_page=True)
            except PlaywrightError:
                pass
            browser.close()


def main(url: str, timeout_s: int = 150) -> int:
    for i in range(1, ATTEMPTS + 1):
        log(f"Attempt {i}/{ATTEMPTS}")
        if attempt(url, timeout_s):
            if os.path.exists(DEBUG_PNG):
                os.remove(DEBUG_PNG)
            return 0
        time.sleep(10)
    print(f"FAILED after {ATTEMPTS} attempts: {url}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
