"""Open the Streamlit app in a headless browser, switch to Paper Mode,
run an episode, and capture screenshots of Figure 1 and Figure 2."""

import time
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT_STATE = "paper/figures/fig_scheduling_state.png"
OUT_GRAPH = "paper/figures/fig_process_graph.png"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        print("Navigating to Streamlit app...")
        page.goto(URL, wait_until="networkidle", timeout=30000)
        time.sleep(3)

        # Switch to Paper Mode
        print("Switching to Paper Mode...")
        paper_radio = page.locator('label:has-text("Paper Mode")')
        if paper_radio.count() > 0:
            paper_radio.click()
            time.sleep(3)
        else:
            print("Paper Mode radio not found, trying alternative...")
            page.get_by_text("Paper Mode").click()
            time.sleep(3)

        # Set seed to 42 and process count to 5
        print("Setting seed=42, processes=5...")
        # Find seed input and set to 42
        seed_inputs = page.locator('[data-testid="stNumberInput"] input')
        if seed_inputs.count() >= 2:
            seed_input = seed_inputs.nth(0)
            seed_input.click(click_count=3)
            seed_input.fill("42")
            time.sleep(0.5)

            proc_input = seed_inputs.nth(1)
            proc_input.click(click_count=3)
            proc_input.fill("5")
            time.sleep(0.5)

        # Click Generate
        print("Clicking Generate...")
        gen_btn = page.locator('button:has-text("Generate")')
        if gen_btn.count() > 0:
            gen_btn.first.click()
            time.sleep(4)

        # Click "Run Episode" to fill the timeline
        print("Running episode...")
        run_btn = page.locator('button:has-text("Run Episode")')
        if run_btn.count() > 0:
            run_btn.first.click()
            time.sleep(5)

        # Wait for page to stabilize
        page.wait_for_load_state("networkidle", timeout=15000)
        time.sleep(3)

        # Screenshot Figure 1: Scheduling State
        print("Capturing Figure 1: Scheduling State...")
        fig1 = page.locator('h2:has-text("FIGURE 1")')
        if fig1.count() > 0:
            container = fig1.locator("xpath=ancestor::div[contains(@class,'stContainer') or @data-testid='stVerticalBlock']").first
            # Get the parent container with border
            parent = fig1.locator("xpath=ancestor::div[@data-testid='stExpander' or contains(@style,'border')]").first
            try:
                parent.screenshot(path=OUT_STATE)
                print(f"  Saved {OUT_STATE}")
            except:
                # Fallback: screenshot a wider area
                bbox = fig1.bounding_box()
                if bbox:
                    page.screenshot(
                        path=OUT_STATE,
                        clip={"x": 0, "y": max(0, bbox["y"] - 20),
                              "width": 1600, "height": 450}
                    )
                    print(f"  Saved {OUT_STATE} (clip fallback)")
        else:
            print("  FIGURE 1 heading not found, trying full page sections...")
            # Take screenshot of the first bordered container after Paper Mode
            containers = page.locator('[data-testid="stVerticalBlockBorderWrapper"]')
            if containers.count() >= 2:
                containers.nth(1).screenshot(path=OUT_STATE)
                print(f"  Saved {OUT_STATE} (container fallback)")

        # Screenshot Figure 2: Dynamic Process Graph
        print("Capturing Figure 2: Dynamic Process Graph...")
        fig2 = page.locator('h2:has-text("FIGURE 2")')
        if fig2.count() > 0:
            parent = fig2.locator("xpath=ancestor::div[@data-testid='stExpander' or contains(@style,'border')]").first
            try:
                parent.screenshot(path=OUT_GRAPH)
                print(f"  Saved {OUT_GRAPH}")
            except:
                bbox = fig2.bounding_box()
                if bbox:
                    page.screenshot(
                        path=OUT_GRAPH,
                        clip={"x": 0, "y": max(0, bbox["y"] - 20),
                              "width": 1600, "height": 500}
                    )
                    print(f"  Saved {OUT_GRAPH} (clip fallback)")
        else:
            print("  FIGURE 2 heading not found, trying containers...")
            containers = page.locator('[data-testid="stVerticalBlockBorderWrapper"]')
            if containers.count() >= 3:
                containers.nth(2).screenshot(path=OUT_GRAPH)
                print(f"  Saved {OUT_GRAPH} (container fallback)")

        # If both failed, take full page and do manual crops
        import os
        if not os.path.exists(OUT_STATE) or os.path.getsize(OUT_STATE) < 1000:
            print("  Falling back to full-page screenshot approach...")
            page.screenshot(path="paper/figures/full_page_debug.png", full_page=True)
            print("  Saved full_page_debug.png for inspection")

        browser.close()
        print("Done!")


if __name__ == "__main__":
    main()
