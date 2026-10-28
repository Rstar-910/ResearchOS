"""Optional browser smoke check: uv run --with playwright python scripts/check_frontend.py.

Uses installed Chrome and an isolated temporary database, never the workspace DB.
"""

import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="researchos-ui-") as directory:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = {
            **os.environ,
            "DATABASE_URL": f"sqlite:///{directory}/browser.db",
            "DISCOVERY_ENABLED": "false",
        }
        server = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/uvicorn"),
                "researchos.api:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(100):
                try:
                    with urlopen(f"{url}/health", timeout=1):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Test server did not start")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="chrome", headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1050})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                expect(page.locator("#welcome")).to_be_visible()
                page.screenshot(path=f"{directory}/welcome.png", full_page=True)
                question = 'How does RAG work? <img src=x onerror="alert(1)">'
                page.locator("#question").fill(question)
                page.locator("#create").click()
                expect(page.locator("#session-title")).to_have_text(question)
                expect(page.locator(".task")).to_have_count(6)
                expect(page.locator("#session-title img")).to_have_count(0)
                page.locator("#tab-sources").click()
                expect(page.locator("#panel")).to_contain_text("No sources discovered yet")
                page.locator("#tab-evidence").click()
                expect(page.locator("#panel")).to_contain_text("Evidence comes before conclusions")
                page.locator("#tab-timeline").click()
                expect(page.locator(".timeline-entry")).to_have_count(3)
                page.locator("#tab-report").click()
                expect(page.locator("#panel")).to_contain_text("Your report will take shape here")
                page.reload()
                expect(page.locator("#session-title")).to_have_text(question)
                page.locator("#refresh-session").click()
                with page.expect_download() as download:
                    page.locator("#export").click()
                assert download.value.suggested_filename.endswith(".json")
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=f"{directory}/session-mobile.png", full_page=True)
                page.on("dialog", lambda dialog: dialog.accept())
                page.locator("#delete").click()
                expect(page.locator("#welcome")).to_be_visible()
                expect(page.locator(".session-link")).to_have_count(0)
                assert not errors, errors
                browser.close()
                print(
                    "Browser check passed: create, tabs, reload, refresh, export, "
                    "mobile, delete, text safety."
                )
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
