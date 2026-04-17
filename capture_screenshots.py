"""
capture_screenshots.py — Capture UI screenshots for README documentation.

Uses Playwright to open the Streamlit app, navigate each tab/component,
and save PNG screenshots to docs/screenshots/.
"""

import time
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = "docs/screenshots"


def wait_for_streamlit(page, timeout=15000):
    """Wait for Streamlit app to finish loading."""
    page.wait_for_load_state("networkidle", timeout=timeout)
    # Wait for the main block container to appear
    page.wait_for_selector(".block-container", timeout=timeout)
    time.sleep(2)  # Extra settle time for CSS/animations


def click_tab(page, tab_label: str):
    """Click a Streamlit tab by its label text."""
    tab = page.locator(f'button[role="tab"]:has-text("{tab_label}")')
    tab.click()
    time.sleep(2)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=2,  # Retina-quality
        )
        page = context.new_page()

        # ── 1. Landing page / Search tab (default) ──
        print("1/8 Landing page...")
        page.goto(URL)
        wait_for_streamlit(page)
        # Expand sidebar
        time.sleep(1)
        page.screenshot(path=f"{OUT}/01_landing_page.png", full_page=True)
        print("  ✓ 01_landing_page.png")

        # ── 2. Sidebar — upload & documents panel ──
        print("2/8 Sidebar...")
        # Take sidebar-focused screenshot by capturing just the sidebar
        sidebar = page.locator('[data-testid="stSidebar"]')
        if sidebar.is_visible():
            sidebar.screenshot(path=f"{OUT}/02_sidebar.png")
            print("  ✓ 02_sidebar.png")
        else:
            # Try clicking the sidebar toggle
            toggle = page.locator('[data-testid="stSidebarCollapsedControl"]')
            if toggle.is_visible():
                toggle.click()
                time.sleep(1)
                sidebar = page.locator('[data-testid="stSidebar"]')
                sidebar.screenshot(path=f"{OUT}/02_sidebar.png")
                print("  ✓ 02_sidebar.png")

        # ── 3. Search tab (already on it) ──
        print("3/8 Search tab...")
        click_tab(page, "Search")
        time.sleep(1)
        page.screenshot(path=f"{OUT}/03_search_tab.png", full_page=True)
        print("  ✓ 03_search_tab.png")

        # ── 4. Chat tab ──
        print("4/8 Chat tab...")
        click_tab(page, "Chat")
        time.sleep(1)
        page.screenshot(path=f"{OUT}/04_chat_tab.png", full_page=True)
        print("  ✓ 04_chat_tab.png")

        # ── 5. Analytics tab ──
        print("5/8 Analytics tab...")
        click_tab(page, "Analytics")
        time.sleep(1)
        page.screenshot(path=f"{OUT}/05_analytics_tab.png", full_page=True)
        print("  ✓ 05_analytics_tab.png")

        # ── 6. Documents tab ──
        print("6/8 Documents tab...")
        click_tab(page, "Documents")
        time.sleep(1)
        page.screenshot(path=f"{OUT}/06_documents_tab.png", full_page=True)
        print("  ✓ 06_documents_tab.png")

        # ── 7. About tab ──
        print("7/8 About tab...")
        click_tab(page, "About")
        time.sleep(1)
        page.screenshot(path=f"{OUT}/07_about_tab.png", full_page=True)
        print("  ✓ 07_about_tab.png")

        # ── 8. Upload flow — upload sample docs and capture indexing ──
        print("8/8 Upload flow...")
        # Go back to search tab first
        click_tab(page, "Search")
        time.sleep(1)

        # Upload sample files via the file uploader
        file_input = page.locator('input[type="file"]')
        if file_input.count() > 0:
            sample_files = [
                "sample_docs/hr_policy.txt",
                "sample_docs/compliance_manual.md",
                "sample_docs/faq.txt",
            ]
            import os
            existing = [f for f in sample_files if os.path.exists(f)]
            if existing:
                file_input.set_input_files(existing)
                time.sleep(2)
                page.screenshot(path=f"{OUT}/08_upload_files.png", full_page=True)
                print("  ✓ 08_upload_files.png")

                # Try clicking the index button
                index_btn = page.locator('button:has-text("Index")')
                if index_btn.count() > 0 and index_btn.first.is_visible():
                    index_btn.first.click()
                    time.sleep(3)
                    page.screenshot(path=f"{OUT}/09_indexing_progress.png", full_page=True)
                    print("  ✓ 09_indexing_progress.png")

                    # Wait for indexing to complete
                    time.sleep(15)
                    page.screenshot(path=f"{OUT}/10_indexing_complete.png", full_page=True)
                    print("  ✓ 10_indexing_complete.png")

                    # Now capture tabs with data
                    # Search tab with documents available
                    click_tab(page, "Search")
                    time.sleep(1)
                    page.screenshot(path=f"{OUT}/11_search_with_docs.png", full_page=True)
                    print("  ✓ 11_search_with_docs.png")

                    # Analytics tab with data
                    click_tab(page, "Analytics")
                    time.sleep(2)
                    page.screenshot(path=f"{OUT}/12_analytics_with_data.png", full_page=True)
                    print("  ✓ 12_analytics_with_data.png")

                    # Documents tab with data
                    click_tab(page, "Documents")
                    time.sleep(1)
                    page.screenshot(path=f"{OUT}/13_documents_explorer.png", full_page=True)
                    print("  ✓ 13_documents_explorer.png")

                    # Try expanding a document
                    expanders = page.locator('details summary')
                    if expanders.count() > 0:
                        expanders.first.click()
                        time.sleep(2)
                        page.screenshot(path=f"{OUT}/14_document_expanded.png", full_page=True)
                        print("  ✓ 14_document_expanded.png")

                    # About tab
                    click_tab(page, "About")
                    time.sleep(1)
                    page.screenshot(path=f"{OUT}/15_about_with_data.png", full_page=True)
                    print("  ✓ 15_about_with_data.png")

                    # Sidebar after indexing
                    sidebar = page.locator('[data-testid="stSidebar"]')
                    if sidebar.is_visible():
                        sidebar.screenshot(path=f"{OUT}/16_sidebar_with_docs.png")
                        print("  ✓ 16_sidebar_with_docs.png")

        browser.close()
        print("\nDone! Screenshots saved to docs/screenshots/")


if __name__ == "__main__":
    main()
