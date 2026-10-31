"""Take proper screenshots of Figure 1 and Figure 2 from the Streamlit Paper Mode."""
import time
from playwright.sync_api import sync_playwright
from PIL import Image

URL = "http://localhost:8501"
OUT_DIR = "/Users/mac4dvc/rl-scheduler-project/.claude/worktrees/xenodochial-mahavira-6bf543/paper/figures"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 2400})

        print("1. Loading app...")
        page.goto(URL, wait_until="networkidle", timeout=30000)
        time.sleep(5)

        print("2. Switch to Paper Mode...")
        page.get_by_text("Paper Mode").click()
        time.sleep(5)

        print("3. Set processes=5...")
        inputs = page.locator('input[type="number"]')
        if inputs.count() >= 2:
            inputs.nth(1).click(click_count=3)
            inputs.nth(1).fill("5")
            inputs.nth(1).press("Enter")
            time.sleep(1)

        print("4. Generate...")
        page.get_by_role("button", name="Generate").click()
        time.sleep(5)

        print("5. Run Episode...")
        page.get_by_role("button", name="Run Episode").click()
        time.sleep(8)
        page.wait_for_load_state("networkidle", timeout=20000)
        time.sleep(3)

        # Take full page screenshot
        fp = f"{OUT_DIR}/debug_fullpage.png"
        page.screenshot(path=fp, full_page=True)
        full = Image.open(fp)
        fw, fh = full.size
        print(f"   Full image: {fw}x{fh}")

        # Use JS to find the bounding boxes of each figure section
        # Streamlit's st.container(border=True) creates a div with border styling
        # Find them by looking for the FIGURE headings and then getting the
        # nearest ancestor with a visible border
        bboxes = page.evaluate("""() => {
            const results = {};
            const headings = document.querySelectorAll('h2');

            for (const h of headings) {
                const text = h.textContent.trim();
                if (text.includes('FIGURE 1') || text.includes('FIGURE 2')) {
                    // Walk up to find a container with border
                    let el = h;
                    let found = null;
                    while (el && el !== document.body) {
                        el = el.parentElement;
                        if (!el) break;
                        const style = window.getComputedStyle(el);
                        // Streamlit containers with border have a visible border
                        if (style.borderWidth && parseFloat(style.borderWidth) > 0
                            && style.borderStyle !== 'none'
                            && el.offsetHeight > 100) {
                            found = el;
                            break;
                        }
                    }

                    if (!found) {
                        // Fallback: go up 5 levels from the heading
                        found = h;
                        for (let i = 0; i < 6 && found.parentElement; i++) {
                            found = found.parentElement;
                        }
                    }

                    const r = found.getBoundingClientRect();
                    results[text.substring(0, 8)] = {
                        x: r.x, y: r.y, width: r.width, height: r.height,
                        tag: found.tagName, classes: found.className.substring(0, 80)
                    };
                }
            }
            return results;
        }""")

        print(f"   Bboxes: {bboxes}")

        for key, bbox in bboxes.items():
            if bbox['height'] < 50:
                print(f"   {key}: too small ({bbox['height']}px), skipping")
                continue

            x = max(0, int(bbox['x']) - 5)
            y = max(0, int(bbox['y']) - 5)
            w = min(fw, int(bbox['x'] + bbox['width']) + 5)
            h = min(fh, int(bbox['y'] + bbox['height']) + 5)

            crop = full.crop((x, y, w, h))

            if "FIGURE 1" in key:
                crop.save(f"{OUT_DIR}/fig_scheduling_state.png")
                print(f"   Fig1: {crop.size}")
            elif "FIGURE 2" in key:
                crop.save(f"{OUT_DIR}/fig_process_graph.png")
                print(f"   Fig2: {crop.size}")

        # If bboxes failed or were too small, use the full page screenshot
        # and manually identify the y-ranges from known pixel positions
        import os
        for name in ["fig_scheduling_state.png", "fig_process_graph.png"]:
            path = f"{OUT_DIR}/{name}"
            if not os.path.exists(path) or os.path.getsize(path) < 5000:
                print(f"   {name} too small or missing, using manual crop from full page")
                # Get all element positions for manual cropping
                positions = page.evaluate("""() => {
                    const all = [];
                    document.querySelectorAll('h2, table, [class*="plotly"], [class*="paper-metric"]').forEach(el => {
                        const r = el.getBoundingClientRect();
                        if (r.height > 10) {
                            all.push({tag: el.tagName, text: el.textContent.substring(0,40),
                                      y: r.y, h: r.height, bottom: r.y + r.height});
                        }
                    });
                    return all.sort((a,b) => a.y - b.y);
                }""")
                for pos in positions:
                    print(f"      {pos['tag']} y={pos['y']:.0f} h={pos['h']:.0f} '{pos['text'][:30]}'")

                # Manual crop based on heading positions
                if name == "fig_scheduling_state.png":
                    # From FIGURE 1 heading to just before FIGURE 2
                    fig1_top = None
                    fig2_top = None
                    for pos in positions:
                        if "FIGURE 1" in pos['text']:
                            fig1_top = int(pos['y']) - 15
                        if "FIGURE 2" in pos['text']:
                            fig2_top = int(pos['y']) - 15
                    if fig1_top and fig2_top:
                        crop = full.crop((40, fig1_top, fw - 40, fig2_top - 10))
                        crop.save(path)
                        print(f"   Manual crop Fig1: {crop.size}")
                else:
                    # From FIGURE 2 heading to FIGURE 3 or end of graph area
                    fig2_top = None
                    fig3_top = None
                    for pos in positions:
                        if "FIGURE 2" in pos['text']:
                            fig2_top = int(pos['y']) - 15
                        if "FIGURE 3" in pos['text']:
                            fig3_top = int(pos['y']) - 15
                    if fig2_top:
                        bottom = fig3_top - 10 if fig3_top else fig2_top + 500
                        crop = full.crop((40, fig2_top, fw - 40, min(fh, bottom)))
                        crop.save(path)
                        print(f"   Manual crop Fig2: {crop.size}")

        browser.close()
        print("Done!")


if __name__ == "__main__":
    main()
