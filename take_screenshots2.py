"""Take actual screenshots from the running Streamlit Paper Mode dashboard."""

import time
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
FIGURES = "paper/figures"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})

        # Step 1: Load
        print("1. Loading Streamlit app...")
        page.goto(URL, wait_until="networkidle", timeout=30000)
        time.sleep(5)

        # Step 2: Full page screenshot to debug
        page.screenshot(path=f"{FIGURES}/debug_01_initial.png", full_page=True)
        print("   Saved debug_01_initial.png")

        # Step 3: Click Paper Mode
        print("2. Switching to Paper Mode...")
        page.get_by_text("Paper Mode").click()
        time.sleep(5)
        page.screenshot(path=f"{FIGURES}/debug_02_paper_mode.png", full_page=True)
        print("   Saved debug_02_paper_mode.png")

        # Step 4: Set processes to 5
        print("3. Setting processes=5...")
        # The paper mode has: Scheduler, Seed, Processes, Generate, Step, Run Episode, Reset
        number_inputs = page.locator('input[type="number"]')
        count = number_inputs.count()
        print(f"   Found {count} number inputs")

        if count >= 2:
            # Second number input is processes
            proc_input = number_inputs.nth(1)
            proc_input.click(click_count=3)
            proc_input.fill("5")
            proc_input.press("Enter")
            time.sleep(1)

        # Step 5: Click Generate
        print("4. Clicking Generate...")
        page.get_by_role("button", name="Generate").click()
        time.sleep(5)
        page.screenshot(path=f"{FIGURES}/debug_03_generated.png", full_page=True)
        print("   Saved debug_03_generated.png")

        # Step 6: Run Episode
        print("5. Running episode...")
        page.get_by_role("button", name="Run Episode").click()
        time.sleep(8)
        page.wait_for_load_state("networkidle", timeout=20000)
        time.sleep(3)

        # Step 7: Full page screenshot after episode
        page.screenshot(path=f"{FIGURES}/debug_04_after_run.png", full_page=True)
        print("   Saved debug_04_after_run.png")

        # Step 8: Now find and screenshot the actual figure containers
        print("6. Capturing individual figures...")

        # Find all bordered containers (Paper Mode wraps figures in st.container(border=True))
        containers = page.locator('[data-testid="stVerticalBlockBorderWrapper"]')
        num_containers = containers.count()
        print(f"   Found {num_containers} bordered containers")

        for i in range(num_containers):
            c = containers.nth(i)
            bbox = c.bounding_box()
            if bbox:
                print(f"   Container {i}: y={bbox['y']:.0f}, h={bbox['height']:.0f}, w={bbox['width']:.0f}")
                c.screenshot(path=f"{FIGURES}/debug_container_{i}.png")

        # Also try finding by heading text
        for heading, outfile in [
            ("FIGURE 1", "fig_scheduling_state.png"),
            ("FIGURE 2", "fig_process_graph.png"),
        ]:
            h = page.locator(f"text={heading}")
            if h.count() > 0:
                print(f"   Found '{heading}' heading")
                # Walk up to the bordered container
                # The structure is: stVerticalBlockBorderWrapper > ... > h2
                wrapper = h.locator("xpath=ancestor::div[@data-testid='stVerticalBlockBorderWrapper']")
                if wrapper.count() > 0:
                    wrapper.first.screenshot(path=f"{FIGURES}/{outfile}")
                    print(f"   Saved {outfile}")
                else:
                    print(f"   Could not find wrapper for {heading}")
            else:
                print(f"   '{heading}' heading NOT found")

        browser.close()
        print("\nDone! Check paper/figures/debug_*.png to see the app state.")


if __name__ == "__main__":
    main()
