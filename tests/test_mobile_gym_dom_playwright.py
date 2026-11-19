"""Optional live-browser check for the MobileGym DOM extractor.

Skipped unless ``MOBILEGYM_RUN_BROWSER_TESTS=1``. The mobilerun venv uses
Playwright 1.48; run ``playwright install chromium`` first.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("playwright.async_api")

pytestmark = pytest.mark.skipif(
    os.environ.get("MOBILEGYM_RUN_BROWSER_TESTS") != "1",
    reason="set MOBILEGYM_RUN_BROWSER_TESTS=1 to run the live DOM test",
)

HTML = """
<!doctype html><html><body>
<div id="root" style="width:360px;height:800px;position:relative;overflow:hidden">
  <button data-trigger="app.save" data-trigger-type="tap"
          style="position:absolute;left:10px;top:20px;width:100px;height:40px">Save</button>
  <button aria-label="Close dialog"
          style="position:absolute;left:150px;top:20px;width:40px;height:40px">X</button>
  <input id="secret" type="password" value="hunter2"
         style="position:absolute;left:10px;top:80px;width:200px;height:30px">
  <div id="vscroll" data-scroll-container="main" data-scroll-direction="vertical"
       style="position:absolute;left:10px;top:180px;width:200px;height:80px;overflow-y:auto">
    <div style="height:400px">tall</div>
  </div>
  <div id="hscroll" data-scroll-container="row" data-scroll-direction="horizontal"
       style="position:absolute;left:10px;top:280px;width:200px;height:50px;overflow-x:auto;overflow-y:hidden">
    <div style="width:600px;height:20px">wide</div>
  </div>
  <iframe srcdoc="<button data-trigger='inner' style='width:80px;height:30px'>Inner</button>"
          title="inner frame"
          style="position:absolute;left:10px;top:350px;width:200px;height:100px;border:0"></iframe>
</div>
<div style="position:fixed;left:250px;top:20px;width:80px;height:40px;z-index:3000">
  <button data-action="portal.close" data-action-type="tap"
          style="width:80px;height:40px">Portal</button>
</div>
</body></html>
"""


def _walk(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in nodes:
        out.append(node)
        out.extend(_walk(node.get("children") or []))
    return out


def _find(nodes: List[Dict[str, Any]], predicate) -> Optional[Dict[str, Any]]:
    for node in _walk(nodes):
        if predicate(node):
            return node
    return None


def test_real_dom_tree_matches_the_portal_contract():
    from playwright.async_api import async_playwright

    from mobilerun.tools.driver.mobile_gym_dom import build_dom_tree

    async def scenario():
        async with async_playwright() as pw:
            try:
                browser = await pw.chromium.launch()
            except Exception as exc:  # noqa: BLE001 - no browser in this env
                pytest.skip(f"chromium unavailable: {exc}")
            try:
                context = await browser.new_context(
                    viewport={"width": 360, "height": 800},
                    device_scale_factor=3,
                )
                page = await context.new_page()
                await page.set_content(HTML)
                await page.wait_for_timeout(250)
                return await build_dom_tree(page, physical_size=(1080, 2400))
            finally:
                await browser.close()

    result = asyncio.run(scenario())
    nodes = result["a11y_tree"]["children"]

    save = _find(nodes, lambda node: node.get("text") == "Save")
    assert save is not None
    assert save["boundsInScreen"] == {
        "left": 54,
        "top": 84,
        "right": 354,
        "bottom": 204,
    }
    assert save["isClickable"] is True
    assert save["resourceId"] == "app.save"

    close = _find(nodes, lambda node: node.get("contentDescription") == "Close dialog")
    assert close is not None

    secret = _find(nodes, lambda node: node.get("resourceId") == "secret")
    assert secret is not None
    assert secret["text"] == ""
    assert secret["isEditable"] is True

    portal = _find(nodes, lambda node: node.get("text") == "Portal")
    assert portal is not None

    vertical = _find(nodes, lambda node: node.get("resourceId") == "vscroll")
    assert vertical is not None
    assert vertical["isScrollable"] is True
    assert vertical["scrollDirection"] == "vertical"

    horizontal = _find(nodes, lambda node: node.get("resourceId") == "hscroll")
    assert horizontal is not None
    assert horizontal["isScrollable"] is False
    assert horizontal["scrollDirection"] == "horizontal"

    webview = _find(nodes, lambda node: node.get("className") == "android.webkit.WebView")
    assert webview is not None
    assert webview["children"]
    assert webview["children"][0]["text"] == "Inner"
