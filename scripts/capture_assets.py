"""Verify every page renders in a real browser, then capture README screenshots + demo GIF.

Usage (app must be running, e.g. `streamlit run app.py --server.port 8599`):
    python scripts/capture_assets.py --url http://localhost:8599
Requires the dev extras: pip install -r requirements-dev.txt && playwright install chromium
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs"
SHOTS = OUT / "screenshots"
PAGES = ["", "BSM_Pricer", "Greeks_Explorer", "Strategy_Builder", "Implied_Vol", "Monte_Carlo", "Binomial_Tree", "Methodology"]


def settle(page: Page, extra: float = 0.9) -> None:
    time.sleep(0.6)
    try:
        page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60000)
    except Exception:
        pass
    time.sleep(extra)


def errors_on(page: Page) -> list[str]:
    found = [e.inner_text() for e in page.locator('[data-testid="stException"]').all()]
    found += [e.inner_text() for e in page.locator('[data-testid="stAlertContainer"]:has-text("Traceback")').all()]
    return found


def choose(page: Page, scope: str, index: int, option: str) -> None:
    page.locator(f'{scope} [data-testid="stSelectbox"]').nth(index).click()
    page.get_by_role("option", name=re.compile(option)).first.click()
    settle(page)


def tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=re.compile(name)).first.click()
    settle(page, 1.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8599")
    ap.add_argument("--no-assets", action="store_true", help="only verify pages")
    args = ap.parse_args()
    SHOTS.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    frames: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium") if Path("/opt/pw-browsers/chromium").exists() else p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page = ctx.new_page()
        console_errors: list[str] = []
        page.on("pageerror", lambda e: console_errors.append(str(e)))

        # ---- 1. render check of every page (default state) -------------------------------------
        for name in PAGES:
            page.goto(f"{args.url}/{name}")
            page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=60000)
            settle(page, 1.5)
            errs = errors_on(page)
            charts = page.locator(".js-plotly-plot").count()
            status = "OK " if not errs else "ERR"
            print(f"[{status}] /{name or '(home)'}: {charts} plotly charts, {len(errs)} exceptions")
            problems += [f"/{name}: {e[:300]}" for e in errs]

        if args.no_assets:
            browser.close()
            return 1 if problems or console_errors else 0

        def shot(fname: str, frame: bool = True) -> None:
            path = SHOTS / fname
            page.screenshot(path=str(path))
            if frame:
                frames.append(path)

        # ---- 2. screenshots + GIF frames ---------------------------------------------------------
        page.goto(f"{args.url}/")
        settle(page, 1.5)
        shot("00_home.png")

        page.goto(f"{args.url}/BSM_Pricer")
        settle(page, 1.5)
        choose(page, '[data-testid="stSidebar"]', 0, "Nifty 50")
        shot("01_bsm_pricer_nifty.png")

        page.goto(f"{args.url}/Greeks_Explorer")
        settle(page, 1.5)
        page.get_by_text("Gamma", exact=True).first.click()
        settle(page, 1.2)
        shot("02_greeks_explorer.png")

        page.goto(f"{args.url}/Strategy_Builder")
        settle(page, 2.0)
        choose(page, '[data-testid="stMain"]', 0, "Iron condor")
        shot("03_strategy_iron_condor.png")

        page.goto(f"{args.url}/Implied_Vol")
        settle(page, 1.5)
        shot("04a_implied_vol_single.png", frame=False)
        tab(page, "Volatility smile")
        shot("04_implied_vol_smile.png")

        page.goto(f"{args.url}/Monte_Carlo")
        settle(page, 3.0)
        shot("05a_mc_european.png", frame=False)
        tab(page, "Convergence")
        shot("05_mc_convergence.png")
        tab(page, "Path fan")
        shot("06_mc_path_fan.png")

        page.goto(f"{args.url}/Binomial_Tree")
        settle(page, 2.5)
        shot("07_binomial_tree.png")
        browser.close()

    # ---- 3. GIF -----------------------------------------------------------------------------------
    imgs = []
    for f in frames:
        im = Image.open(f).convert("RGB")
        w = 960
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
        imgs.append(im)
    h = max(i.height for i in imgs)
    pal = [i.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for i in imgs]
    pal[0].save(OUT / "demo.gif", save_all=True, append_images=pal[1:], duration=1800, loop=0, optimize=True)
    print(f"wrote {len(frames)}-frame GIF: {(OUT / 'demo.gif').stat().st_size / 1e6:.2f} MB")

    if console_errors:
        print("JS page errors:", console_errors[:5])
    if problems:
        print("PROBLEMS:\n" + "\n".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
