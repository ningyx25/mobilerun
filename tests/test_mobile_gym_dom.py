"""DOM-derived MobileGym a11y tree: composition, merging and failure isolation.

No browser is started here; the Playwright page/frame surface is faked so the
coordinate math and the portal contract can be checked deterministically.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from mobilerun.tools.driver.mobile_gym_dom import (
    FramePlacement,
    Rect,
    Transform,
    build_dom_tree,
    clip_tree,
    compose_transform,
    find_iframe_node,
    physical_bounds,
)


def run(coro):
    return asyncio.run(coro)


def _leaf(bounds: Dict[str, float], **overrides: Any) -> Dict[str, Any]:
    node: Dict[str, Any] = {
        "tag": "button",
        "role": "",
        "text": "OK",
        "content_description": "",
        "hint_text": "",
        "resource_id": "",
        "bounds": dict(bounds),
        "visible_bounds": dict(bounds),
        "clickable": True,
        "long_clickable": False,
        "editable": False,
        "checkable": False,
        "checked": False,
        "selected": False,
        "scrollable": False,
        "scroll_direction": "none",
        "focused": False,
        "enabled": True,
        "visible_to_user": True,
        "frame_ordinal": -1,
        "frame_url": "",
        "title": "",
        "children": [],
    }
    node.update(overrides)
    return node


def _structural(children: List[Dict[str, Any]], **overrides: Any) -> Dict[str, Any]:
    node: Dict[str, Any] = {
        "structural": True,
        "tag": "div",
        "children": children,
    }
    node.update(overrides)
    return node


def _payload(
    nodes: List[Dict[str, Any]],
    *,
    viewport: tuple[float, float] = (360, 800),
    dpr: float = 3.0,
    phone: Optional[Dict[str, Any]] = None,
    root: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "tree": root if root is not None else _structural(nodes),
        "viewport": {"width": viewport[0], "height": viewport[1]},
        "dpr": dpr,
        "phone_state": phone or {},
        "truncated": False,
    }


def _placement(
    *,
    ordinal: int = 0,
    rect: tuple[float, float, float, float] = (100, 200, 180, 400),
    client: tuple[float, float] = (180, 400),
) -> Dict[str, Any]:
    return {
        "ordinal": ordinal,
        "rect": {
            "x": rect[0],
            "y": rect[1],
            "width": rect[2],
            "height": rect[3],
        },
        "client_width": client[0],
        "client_height": client[1],
        "border_left": 0,
        "border_top": 0,
        "border_right": 0,
        "border_bottom": 0,
        "padding_left": 0,
        "padding_top": 0,
        "padding_right": 0,
        "padding_bottom": 0,
        "display": "block",
        "visibility": "visible",
        "opacity": 1.0,
        "src": "about:blank",
        "title": "frame",
    }


class FakeHandle:
    def __init__(self, info: Dict[str, Any]) -> None:
        self.info = info

    async def evaluate(self, _script: str) -> Dict[str, Any]:
        return self.info


class FakeFrame:
    def __init__(
        self,
        payload: Any,
        *,
        parent: Optional["FakeFrame"] = None,
        placement: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.payload = payload
        self.parent_frame = parent
        self.child_frames: List["FakeFrame"] = []
        self.placement = placement

    async def evaluate(self, _script: str, _options: Any = None) -> Any:
        if self.payload is None:
            raise RuntimeError("frame detached")
        return self.payload

    async def frame_element(self) -> FakeHandle:
        if self.placement is None:
            raise RuntimeError("frame detached")
        return FakeHandle(self.placement)


class FakePage:
    def __init__(self, main_frame: FakeFrame) -> None:
        self.main_frame = main_frame


def _iframe(bounds: Dict[str, float], ordinal: int = 0) -> Dict[str, Any]:
    return _leaf(
        bounds,
        tag="iframe",
        text="",
        clickable=False,
        content_description="frame",
        resource_id="frame",
        frame_ordinal=ordinal,
        frame_url="about:blank",
        title="frame",
    )


def test_main_frame_bounds_are_physical_pixels():
    main = FakeFrame(_payload([_leaf({"x": 10, "y": 20, "width": 30, "height": 40})]))

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    node = result["a11y_tree"]["children"][0]
    assert node["boundsInScreen"] == {
        "left": 30,
        "top": 60,
        "right": 120,
        "bottom": 180,
    }
    assert node["className"] == "android.widget.Button"
    assert node["enabled"] is True
    assert node["isEnabled"] is True


def test_iframe_content_is_offset_and_scaled():
    main = FakeFrame(_payload([_iframe({"x": 100, "y": 200, "width": 180, "height": 400})]))
    child = FakeFrame(
        _payload([_leaf({"x": 0, "y": 0, "width": 100, "height": 50})], viewport=(360, 800)),
        parent=main,
        placement=_placement(ordinal=0, rect=(100, 200, 180, 400), client=(180, 400)),
    )
    main.child_frames.append(child)

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    webview = result["a11y_tree"]["children"][0]
    assert webview["className"] == "android.webkit.WebView"
    assert webview["children"][0]["boundsInScreen"] == {
        "left": 300,
        "top": 600,
        "right": 450,
        "bottom": 675,
    }


def test_nested_iframes_compose_parent_offsets():
    main = FakeFrame(_payload([_iframe({"x": 100, "y": 200, "width": 180, "height": 400})]))
    child = FakeFrame(
        _payload([_iframe({"x": 20, "y": 40, "width": 100, "height": 100})], viewport=(360, 800)),
        parent=main,
        placement=_placement(ordinal=0, rect=(100, 200, 180, 400), client=(180, 400)),
    )
    grandchild = FakeFrame(
        _payload([_leaf({"x": 0, "y": 0, "width": 10, "height": 10})], viewport=(100, 100)),
        parent=child,
        placement=_placement(ordinal=0, rect=(20, 40, 100, 100), client=(100, 100)),
    )
    main.child_frames.append(child)
    child.child_frames.append(grandchild)

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    outer = result["a11y_tree"]["children"][0]
    inner = outer["children"][0]
    leaf = inner["children"][0]
    assert leaf["boundsInScreen"] == {
        "left": 330,
        "top": 660,
        "right": 345,
        "bottom": 675,
    }


def test_child_frame_attaches_to_the_matching_ordinal():
    main = FakeFrame(
        _payload(
            [
                _iframe({"x": 0, "y": 0, "width": 100, "height": 100}, ordinal=0),
                _iframe({"x": 100, "y": 200, "width": 180, "height": 400}, ordinal=1),
            ]
        )
    )
    child = FakeFrame(
        _payload([_leaf({"x": 0, "y": 0, "width": 10, "height": 10})], viewport=(360, 800)),
        parent=main,
        placement=_placement(ordinal=1, rect=(100, 200, 180, 400), client=(180, 400)),
    )
    main.child_frames.append(child)

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    first, second = result["a11y_tree"]["children"]
    assert "children" not in first or not first["children"]
    assert second["frameOrdinal"] == 1
    assert second["children"]


def test_unmatched_iframe_gets_a_synthetic_webview():
    main = FakeFrame(_payload([_leaf({"x": 0, "y": 0, "width": 10, "height": 10})]))
    child = FakeFrame(
        _payload([_leaf({"x": 0, "y": 0, "width": 10, "height": 10})], viewport=(360, 800)),
        parent=main,
        placement=_placement(ordinal=0, rect=(100, 200, 180, 400), client=(180, 400)),
    )
    main.child_frames.append(child)

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    webview = result["a11y_tree"]["children"][-1]
    assert webview["className"] == "android.webkit.WebView"
    assert webview["frameOrdinal"] == 0
    assert webview["children"]


def test_hidden_iframe_subtree_is_not_attached():
    main = FakeFrame(
        _payload([_iframe({"x": 100, "y": 200, "width": 180, "height": 400})])
    )
    main.payload["tree"]["children"][0]["visible_to_user"] = False
    child = FakeFrame(
        _payload([_leaf({"x": 0, "y": 0, "width": 10, "height": 10})], viewport=(360, 800)),
        parent=main,
        placement=_placement(ordinal=0, rect=(100, 200, 180, 400), client=(180, 400)),
    )
    main.child_frames.append(child)

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    webview = result["a11y_tree"]["children"][0]
    assert not webview.get("children")


def test_detached_frame_is_skipped_without_failing():
    main = FakeFrame(_payload([_leaf({"x": 0, "y": 0, "width": 10, "height": 10})]))
    child = FakeFrame(_payload([]), parent=main, placement=None)
    main.child_frames.append(child)

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    assert result["a11y_tree"]["children"]


def test_malformed_child_payload_is_skipped():
    main = FakeFrame(_payload([_leaf({"x": 0, "y": 0, "width": 10, "height": 10})]))
    child = FakeFrame(True, parent=main, placement=_placement())
    main.child_frames.append(child)

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    assert len(result["a11y_tree"]["children"]) == 1


def test_total_node_cap_is_deterministic():
    nodes = [
        _leaf({"x": index, "y": 0, "width": 10, "height": 10}) for index in range(5)
    ]
    main = FakeFrame(_payload(nodes))

    result = run(
        build_dom_tree(
            FakePage(main), physical_size=(1080, 2400), max_total_nodes=3
        )
    )

    assert len(result["a11y_tree"]["children"]) == 3
    assert result["truncated"] is True


def test_phone_state_and_package_name_follow_the_active_app():
    phone = {
        "currentApp": "wechat",
        "packageName": "wechat",
        "focusedElement": {"text": "Ada"},
        "isEditable": True,
    }
    main = FakeFrame(_payload([_leaf({"x": 0, "y": 0, "width": 10, "height": 10})], phone=phone))

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    assert result["phone_state"] == phone
    assert result["a11y_tree"]["packageName"] == "wechat"
    assert result["a11y_tree"]["children"][0]["packageName"] == "wechat"


def test_portal_contract_fields_are_emitted():
    main = FakeFrame(
        _payload(
            [
                _leaf(
                    {"x": 0, "y": 0, "width": 10, "height": 10},
                    hint_text="name",
                    resource_id="app.save",
                    checkable=True,
                    checked=True,
                    selected=True,
                    scrollable=True,
                    scroll_direction="vertical",
                )
            ]
        )
    )

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    node = result["a11y_tree"]["children"][0]
    for key in (
        "boundsInScreen",
        "className",
        "text",
        "contentDescription",
        "resourceId",
        "packageName",
        "hintText",
        "isCheckable",
        "isChecked",
        "isClickable",
        "isLongClickable",
        "isEditable",
        "isFocused",
        "isVisibleToUser",
        "isScrollable",
        "scrollDirection",
        "isSelected",
        "isEnabled",
        "enabled",
    ):
        assert key in node
    assert node["isCheckable"] is True
    assert node["isChecked"] is True
    assert node["isSelected"] is True
    assert node["isScrollable"] is True
    assert node["scrollDirection"] == "vertical"


def test_multiple_top_level_roots_are_kept():
    main = FakeFrame(
        _payload(
            [
                _structural([_leaf({"x": 0, "y": 0, "width": 10, "height": 10})]),
                _leaf({"x": 20, "y": 20, "width": 10, "height": 10}, text="Portal"),
            ]
        )
    )

    result = run(build_dom_tree(FakePage(main), physical_size=(1080, 2400)))

    assert len(result["a11y_tree"]["children"]) == 2


def test_compose_transform_handles_padding_and_borders():
    placement = FramePlacement(
        ordinal=0,
        rect=Rect(10, 20, 200, 100),
        client_width=200,
        client_height=102,
        border_left=2,
        border_top=3,
        border_right=2,
        border_bottom=3,
        padding_left=4,
        padding_top=5,
        padding_right=4,
        padding_bottom=5,
        child_viewport=(192, 92),
    )

    transform = compose_transform(Transform(0, 0, 1, 1), placement)

    assert transform.ox == 16
    assert transform.oy == 28
    assert transform.sx == 1
    assert transform.sy == 1


def test_physical_bounds_clip_to_the_screen():
    bounds = physical_bounds(
        {"x": -10, "y": -10, "width": 20, "height": 20},
        Transform(0, 0, 1, 1),
        3,
        3,
        (1080, 2400),
    )
    assert bounds == {"left": 0, "top": 0, "right": 30, "bottom": 30}


def test_clip_tree_drops_nodes_outside_the_clip():
    nodes = [
        {
            "boundsInScreen": {"left": 0, "top": 0, "right": 10, "bottom": 10},
            "children": [],
        },
        {
            "boundsInScreen": {"left": 100, "top": 100, "right": 110, "bottom": 110},
            "children": [],
        },
    ]

    clipped = clip_tree(nodes, {"left": 0, "top": 0, "right": 50, "bottom": 50})

    assert len(clipped) == 1


def test_find_iframe_node_searches_children():
    tree = [{"frameOrdinal": 0, "children": [{"frameOrdinal": 1, "children": []}]}]

    assert find_iframe_node(tree, 1)["frameOrdinal"] == 1
    assert find_iframe_node(tree, 2) is None
