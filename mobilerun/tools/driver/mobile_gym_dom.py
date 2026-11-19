"""DOM-derived, portal-shaped accessibility tree for MobileGym.

MobileGym is a React DOM application, not an Android device: there is no
uiautomator tree. This module synthesises the same *portal* contract the
AndroidWorld and MobileWorld drivers expose -- ``boundsInScreen`` in physical
screenshot pixels plus the Android-style node flags -- from the rendered DOM,
so a tree-based policy (``jev-mobile``) can consume it unchanged.

The extractor is deliberately conservative:

* it only runs when the backend URL carries ``?ui_tree=dom``;
* every frame (including cross-origin iframes) is sampled through Playwright
  and its coordinates are composed into the main frame's viewport;
* failures are isolated per frame; the caller falls back to an empty stub.

The DOM is not a native accessibility tree. See the project plan for the
coverage boundary: un-tagged ``div onClick`` controls, canvas/map content and
actions inside iframes are not represented.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("mobilerun")

#: Per-frame caps. The agent additionally renders at most 400 tree lines and
#: offers at most 120 candidates, so these are only about payload sanity.
MAX_SCAN_ELEMENTS = 20000
MAX_NODES_PER_FRAME = 1200
MAX_TOTAL_NODES = 2000
MAX_DEPTH = 40
MAX_FRAMES = 24
FRAME_TIMEOUT_S = 2.0
TOTAL_TIMEOUT_S = 8.0
MAX_LABEL_CHARS = 200

#: The frame-local extractor. ``opts``:
#: ``rootSelector`` (``"body"``), ``isMainFrame``, ``maxNodes``,
#: ``maxDepth``, ``maxScan``, ``maxLabelChars``.
EXTRACTOR_JS = r"""
(opts) => {
  const MAX_NODES = opts.maxNodes || 1200;
  const MAX_DEPTH = opts.maxDepth || 40;
  const MAX_SCAN = opts.maxScan || 20000;
  const MAX_LABEL = opts.maxLabelChars || 200;
  const rootEl =
    (opts.rootSelector && document.querySelector(opts.rootSelector)) ||
    document.body;
  const viewport = { width: window.innerWidth, height: window.innerHeight };
  const empty = {
    tree: null,
    viewport,
    dpr: window.devicePixelRatio || 1,
    phone_state: {},
    truncated: false,
    node_count: 0,
  };
  if (!rootEl) return empty;

  const SKIP_TAGS = new Set([
    "script", "style", "noscript", "template", "head", "meta", "link",
  ]);
  const INTERACTIVE_ROLES = new Set([
    "button", "link", "tab", "checkbox", "radio", "switch", "option",
    "menuitem", "textbox", "searchbox", "combobox",
  ]);
  const TEXT_INPUT_TYPES = new Set([
    "text", "search", "email", "url", "tel", "password", "number",
    "date", "time", "datetime-local", "month", "week",
  ]);
  const CLIPPING_OVERFLOW = new Set(["hidden", "clip", "auto", "scroll"]);
  const CONTROL_SELECTOR = [
    "button", "a[href]", "input", "textarea", "select", "summary",
    "[contenteditable]:not([contenteditable='false'])", "[data-trigger]",
    "[data-action]", "[data-scroll-container]", "iframe",
    "[role='button']", "[role='link']", "[role='tab']", "[role='checkbox']",
    "[role='radio']", "[role='switch']", "[role='option']",
    "[role='menuitem']", "[role='textbox']", "[role='searchbox']",
    "[role='combobox']",
  ].join(",");

  const norm = (value) =>
    String(value || "").replace(/\s+/g, " ").trim().slice(0, MAX_LABEL);
  const attr = (el, name) =>
    (el && el.getAttribute ? el.getAttribute(name) || "" : "");
  const cls = (el) =>
    typeof el.className === "string" ? el.className : "";

  const isEditableEl = (el) => {
    const tag = (el.tagName || "").toLowerCase();
    if (tag === "textarea") return !el.readOnly && !el.disabled;
    if (tag === "input") {
      const type = (el.type || "text").toLowerCase();
      return TEXT_INPUT_TYPES.has(type) && !el.readOnly && !el.disabled;
    }
    if (
      el.getAttribute &&
      el.getAttribute("contenteditable") !== null &&
      el.getAttribute("contenteditable") !== "false"
    ) {
      return !el.disabled;
    }
    const role = (attr(el, "role") || "").toLowerCase();
    return role === "textbox" || role === "searchbox";
  };

  const isCheckableEl = (el) => {
    const tag = (el.tagName || "").toLowerCase();
    const type = (el.type || "").toLowerCase();
    const role = (attr(el, "role") || "").toLowerCase();
    return (
      (tag === "input" && (type === "checkbox" || type === "radio")) ||
      ["checkbox", "radio", "switch"].includes(role) ||
      el.hasAttribute("aria-checked")
    );
  };

  const triggerType = (el) =>
    (attr(el, "data-trigger-type") || attr(el, "data-action-type") || "")
      .toLowerCase();
  const hasBinding = (el) =>
    el.hasAttribute("data-trigger") || el.hasAttribute("data-action");
  const isLongPress = (el) => triggerType(el) === "longpress";
  const isPointerCursor = (el) => cls(el).split(/\s+/).includes("cursor-pointer");

  const isSemanticClickable = (el) => {
    const tag = (el.tagName || "").toLowerCase();
    const type = (el.type || "").toLowerCase();
    const role = (attr(el, "role") || "").toLowerCase();
    if (tag === "button") return true;
    if (tag === "a" && el.hasAttribute("href")) return true;
    if (
      tag === "input" &&
      ["button", "submit", "reset", "checkbox", "radio"].includes(type)
    ) {
      return true;
    }
    if (tag === "select" || tag === "summary") return true;
    if (!INTERACTIVE_ROLES.has(role)) return false;
    return role !== "textbox" && role !== "searchbox";
  };

  const isClickableEl = (el) =>
    (hasBinding(el) && !isLongPress(el)) ||
    isSemanticClickable(el) ||
    isPointerCursor(el);

  const isTextControl = (el) => {
    const tag = (el.tagName || "").toLowerCase();
    return (
      isClickableEl(el) ||
      isLongPress(el) ||
      isEditableEl(el) ||
      isCheckableEl(el) ||
      tag === "iframe"
    );
  };

  const isControlEl = (el) => {
    const tag = (el.tagName || "").toLowerCase();
    return (
      isTextControl(el) ||
      tag === "iframe" ||
      el.hasAttribute("data-scroll-container")
    );
  };

  const hasScrollClass = (el) =>
    /(^|\s)(overflow-(y-)?(auto|scroll)|overflow-hidden)(\s|$)/.test(cls(el));

  const isInteresting = (el) => {
    const tag = (el.tagName || "").toLowerCase();
    if (tag === "iframe") return true;
    if (tag === "img") return !!attr(el, "alt");
    if (isControlEl(el)) return true;
    if (hasScrollClass(el)) return true;
    if (["button", "a", "input", "textarea", "select", "summary", "label"].includes(tag)) {
      if (tag === "a" && !el.hasAttribute("href")) return false;
      if (tag === "input" && (el.type || "").toLowerCase() === "hidden") return false;
      return true;
    }
    const role = (attr(el, "role") || "").toLowerCase();
    if (INTERACTIVE_ROLES.has(role)) return true;
    if (
      attr(el, "aria-label") ||
      attr(el, "aria-labelledby") ||
      attr(el, "title") ||
      attr(el, "placeholder")
    ) {
      return true;
    }
    if (el.hasAttribute("tabindex") && Number(el.getAttribute("tabindex")) >= 0) {
      return true;
    }
    for (const node of el.childNodes) {
      if (node.nodeType === 3 && norm(node.textContent)) return true;
    }
    return false;
  };

  // -- pass 1: cheap scan, mark interesting nodes and their ancestors --------
  const all = [rootEl].concat(Array.from(rootEl.querySelectorAll("*")));
  const keep = new Set();
  for (let i = 0; i < all.length && i < MAX_SCAN; i++) {
    const el = all[i];
    if (!(el instanceof Element)) continue;
    if (SKIP_TAGS.has(el.tagName.toLowerCase())) continue;
    if (!isInteresting(el)) continue;
    let cur = el;
    let depth = 0;
    while (cur && depth <= MAX_DEPTH) {
      keep.add(cur);
      if (cur === rootEl) break;
      cur = cur.parentElement;
      depth += 1;
    }
  }

  // -- pass 2: style/rect only for kept nodes -------------------------------
  const styles = new Map();
  const styleOf = (el) => {
    let style = styles.get(el);
    if (!style) {
      style = getComputedStyle(el);
      styles.set(el, style);
    }
    return style;
  };

  const intersect = (a, b) => {
    const left = Math.max(a.x, b.x);
    const top = Math.max(a.y, b.y);
    const right = Math.min(a.x + a.width, b.x + b.width);
    const bottom = Math.min(a.y + a.height, b.y + b.height);
    if (right <= left || bottom <= top) return null;
    return { x: left, y: top, width: right - left, height: bottom - top };
  };

  const viewportRect = { x: 0, y: 0, width: viewport.width, height: viewport.height };
  const visibleRect = (el, rect) => {
    let result = intersect(rect, viewportRect);
    if (!result) return null;
    // Do not clip against the root (``body``): ``#root`` is positioned, so
    // body/html can collapse to zero height while the viewport is the real
    // clip. Every real container between the element and body still clips.
    let cur = el.parentElement;
    while (cur && cur !== rootEl) {
      const style = styleOf(cur);
      if (
        CLIPPING_OVERFLOW.has(style.overflowX) ||
        CLIPPING_OVERFLOW.has(style.overflowY)
      ) {
        const box = cur.getBoundingClientRect();
        result = intersect(result, {
          x: box.x,
          y: box.y,
          width: box.width,
          height: box.height,
        });
        if (!result) return null;
      }
      cur = cur.parentElement;
    }
    return result;
  };

  let hitTests = 0;
  const isHitVisible = (el, rect) => {
    if (
      !isClickableEl(el) &&
      !isLongPress(el) &&
      !isEditableEl(el) &&
      !isCheckableEl(el)
    ) {
      return true;
    }
    if (hitTests >= 250) return true;
    const points = [
      [0.5, 0.5],
      [0.2, 0.2],
      [0.8, 0.8],
    ];
    let checked = 0;
    for (const [fx, fy] of points) {
      const x = rect.x + rect.width * fx;
      const y = rect.y + rect.height * fy;
      if (x < 0 || y < 0 || x >= viewport.width || y >= viewport.height) {
        continue;
      }
      hitTests += 1;
      checked += 1;
      const hit = document.elementFromPoint(x, y);
      if (!hit) continue;
      if (el.contains(hit) || hit.contains(el)) return true;
    }
    return checked === 0;
  };

  const labelledByText = (el) => {
    const ids = String(attr(el, "aria-labelledby") || "")
      .split(/\s+/)
      .filter(Boolean);
    const parts = [];
    for (const id of ids) {
      const node = document.getElementById(id);
      if (node) parts.push(node.textContent || "");
    }
    return norm(parts.join(" "));
  };

  const labelText = (el) => {
    const labels = el.labels;
    if (labels && labels.length) return norm(labels[0].textContent || "");
    return "";
  };

  const directText = (el) => {
    let out = "";
    for (const node of el.childNodes) {
      if (node.nodeType === 3) out += node.textContent;
    }
    return norm(out);
  };

  const controlText = (el) => {
    let out = "";
    const walk = (node) => {
      for (const child of node.childNodes) {
        if (child.nodeType === 3) {
          out += child.textContent;
        } else if (child.nodeType === 1) {
          if (child !== el && isControlEl(child)) continue;
          walk(child);
        }
      }
    };
    walk(el);
    return norm(out);
  };

  const hasControlDescendant = (el) => {
    try {
      return el.matches(CONTROL_SELECTOR) || !!el.querySelector(CONTROL_SELECTOR);
    } catch (e) {
      return false;
    }
  };

  const scrollInfo = (el, style) => {
    const canY =
      CLIPPING_OVERFLOW.has(style.overflowY) &&
      el.scrollHeight > el.clientHeight + 2;
    const canX =
      CLIPPING_OVERFLOW.has(style.overflowX) &&
      el.scrollWidth > el.clientWidth + 2;
    const declared = String(attr(el, "data-scroll-direction") || "").toLowerCase();
    let direction = "none";
    if (canY && canX) direction = "both";
    else if (canY) direction = "vertical";
    else if (canX) direction = "horizontal";
    else if (declared === "vertical" || declared === "horizontal") {
      direction = declared;
    }
    const scrollable =
      (direction === "vertical" || direction === "both") && canY;
    return { scrollable, direction };
  };

  const iframeOrdinals = new Map(
    Array.from(document.querySelectorAll("iframe")).map((el, index) => [el, index])
  );

  const describe = (el, style) => {
    const tag = (el.tagName || "").toLowerCase();
    const type = (el.type || "").toLowerCase();
    const editable = isEditableEl(el);
    const checkable = isCheckableEl(el);
    const clickable = isClickableEl(el);
    const longClickable = isLongPress(el);
    const scroll = scrollInfo(el, style);

    let text = "";
    if (editable && tag === "input") {
      text = type === "password" ? "" : norm(el.value);
    } else if (editable && tag === "textarea") {
      text = norm(el.value);
    } else if (editable) {
      text = norm(el.textContent);
    } else if (tag === "select") {
      const option = el.options ? el.options[el.selectedIndex] : null;
      text = norm(option ? option.text : "");
    } else if (clickable || longClickable || checkable) {
      text = controlText(el);
    } else {
      text = directText(el);
    }

    const described =
      labelledByText(el) ||
      norm(attr(el, "aria-label") || "") ||
      (tag === "img" ? norm(attr(el, "alt") || "") : "") ||
      norm(attr(el, "title") || "") ||
      labelText(el);
    const hint = editable
      ? norm(attr(el, "placeholder") || attr(el, "aria-placeholder") || "")
      : "";
    const resource =
      attr(el, "data-trigger") || attr(el, "data-action") || attr(el, "id") || "";
    const control =
      clickable ||
      longClickable ||
      editable ||
      checkable ||
      tag === "iframe" ||
      el.hasAttribute("data-scroll-container");

    if (!control && !text && !described && !hint && !scroll.scrollable && tag !== "iframe") {
      return null;
    }

    const bounds = el.getBoundingClientRect();
    const visible = visibleRect(el, bounds);
    if (!visible) return null;

    return {
      tag,
      role: attr(el, "role") || "",
      text,
      content_description: described,
      hint_text: hint,
      resource_id: resource,
      bounds: {
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
      },
      visible_bounds: {
        x: visible.x,
        y: visible.y,
        width: visible.width,
        height: visible.height,
      },
      clickable,
      long_clickable: longClickable,
      editable,
      checkable,
      checked:
        el.checked === true || attr(el, "aria-checked") === "true",
      selected:
        el.selected === true || attr(el, "aria-selected") === "true",
      scrollable: scroll.scrollable,
      scroll_direction: scroll.direction,
      focused: document.activeElement === el,
      enabled: !el.disabled && attr(el, "aria-disabled") !== "true",
      visible_to_user: isHitVisible(el, visible),
      frame_ordinal: tag === "iframe" ? iframeOrdinals.get(el) ?? -1 : -1,
      frame_url: tag === "iframe" ? attr(el, "src") : "",
      title: tag === "iframe" ? attr(el, "title") : "",
    };
  };

  let nodeCount = 0;
  let truncated = false;
  const build = (el, depth) => {
    if (depth > MAX_DEPTH) return null;
    if (SKIP_TAGS.has(el.tagName.toLowerCase())) return null;
    if (
      el.getAttribute("aria-hidden") === "true" ||
      el.hasAttribute("inert") ||
      el.hasAttribute("hidden")
    ) {
      return null;
    }
    const style = styleOf(el);
    if (style.display === "none" || parseFloat(style.opacity) < 0.01) {
      return null;
    }
    if (style.visibility === "hidden" || style.visibility === "collapse") {
      return null;
    }

    // Only true controls suppress their text-only descendants; a scroll
    // container still needs its list text in the tree.
    const control = isTextControl(el);
    const children = [];
    for (const child of el.children) {
      if (!keep.has(child)) continue;
      if (control && !hasControlDescendant(child)) continue;
      const built = build(child, depth + 1);
      if (built) children.push(built);
    }

    const info = keep.has(el) ? describe(el, style) : null;
    if (!info) {
      if (children.length === 0) return null;
      if (children.length === 1) return children[0];
      return { structural: true, tag: el.tagName.toLowerCase(), children };
    }
    if (nodeCount >= MAX_NODES) {
      truncated = true;
      return null;
    }
    nodeCount += 1;
    info.children = children;
    return info;
  };

  const readPhoneState = () => {
    try {
      const os = window.__OS__ || {};
      let state = {};
      try {
        state = os.state && typeof os.state === "object" ? os.state : {};
      } catch (e) {
        state = {};
      }
      if (!state || Object.keys(state).length === 0) {
        try {
          state = (typeof os.getState === "function" && os.getState()) || {};
        } catch (e) {
          state = {};
        }
      }
      const launcher = !!state.isLauncherVisible;
      const recents = !!state.isRecentsVisible;
      const app = launcher
        ? "launcher"
        : recents
        ? "recents"
        : String(state.activeAppId || "");
      let activity = null;
      try {
        activity = os.getAppRoute ? os.getAppRoute(state.activeAppId)?.path ?? null : null;
      } catch (e) {
        activity = null;
      }
      const active = document.activeElement;
      const editable = !!(active && isEditableEl(active));
      const focusedText = editable
        ? norm(active.value || attr(active, "aria-label") || attr(active, "placeholder"))
        : "";
      return {
        currentApp: app,
        packageName: app,
        activity,
        focusedElement: focusedText ? { text: focusedText } : null,
        isEditable: editable,
        isLauncherVisible: launcher,
        isRecentsVisible: recents,
      };
    } catch (e) {
      return {};
    }
  };

  const tree = build(rootEl, 0);
  return {
    tree: tree || { structural: true, tag: "body", children: [] },
    viewport,
    dpr: window.devicePixelRatio || 1,
    phone_state: opts.isMainFrame ? readPhoneState() : {},
    truncated,
    node_count: nodeCount,
  };
}
"""

#: Runs in the parent frame to describe one iframe element. ``getBoundingClientRect``
#: is evaluated in the parent document, so composition never depends on how
#: Playwright normalises ``bounding_box()`` for child frames.
PLACEMENT_JS = r"""
(el) => {
  const rect = el.getBoundingClientRect();
  const style = getComputedStyle(el);
  const px = (value) => parseFloat(value) || 0;
  return {
    ordinal: Array.from(el.ownerDocument.querySelectorAll("iframe")).indexOf(el),
    rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
    client_width: el.clientWidth,
    client_height: el.clientHeight,
    border_left: px(style.borderLeftWidth),
    border_top: px(style.borderTopWidth),
    border_right: px(style.borderRightWidth),
    border_bottom: px(style.borderBottomWidth),
    padding_left: px(style.paddingLeft),
    padding_top: px(style.paddingTop),
    padding_right: px(style.paddingRight),
    padding_bottom: px(style.paddingBottom),
    display: style.display,
    visibility: style.visibility,
    opacity: px(style.opacity),
    src: el.getAttribute("src") || "",
    title: el.getAttribute("title") || "",
  };
}
"""


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Transform:
    """Maps a frame-local CSS point into the main frame's CSS space."""

    ox: float
    oy: float
    sx: float
    sy: float


@dataclass(frozen=True)
class FramePlacement:
    ordinal: int
    rect: Rect
    client_width: float
    client_height: float
    border_left: float
    border_top: float
    border_right: float
    border_bottom: float
    padding_left: float
    padding_top: float
    padding_right: float
    padding_bottom: float
    child_viewport: Tuple[float, float]
    display: str = "block"
    visibility: str = "visible"
    opacity: float = 1.0
    src: str = ""
    title: str = ""

    def visible(self) -> bool:
        return (
            self.display != "none"
            and self.visibility not in ("hidden", "collapse")
            and self.opacity > 0.01
            and self.rect.width > 0
            and self.rect.height > 0
        )


def compose_transform(parent: Transform, placement: FramePlacement) -> Transform:
    """Compose a child frame's local CSS into the main frame's CSS space."""
    content_width = (
        placement.client_width - placement.padding_left - placement.padding_right
    )
    content_height = (
        placement.client_height - placement.padding_top - placement.padding_bottom
    )
    if content_width <= 0:
        content_width = (
            placement.rect.width - placement.border_left - placement.border_right
        )
    if content_height <= 0:
        content_height = (
            placement.rect.height - placement.border_top - placement.border_bottom
        )
    viewport_width, viewport_height = placement.child_viewport
    if (
        content_width <= 0
        or content_height <= 0
        or viewport_width <= 0
        or viewport_height <= 0
    ):
        raise ValueError("iframe has no content box or child viewport")
    return Transform(
        ox=parent.ox
        + parent.sx * (placement.rect.x + placement.border_left + placement.padding_left),
        oy=parent.oy
        + parent.sy * (placement.rect.y + placement.border_top + placement.padding_top),
        sx=parent.sx * (content_width / viewport_width),
        sy=parent.sy * (content_height / viewport_height),
    )


def physical_bounds(
    local: Dict[str, float],
    transform: Transform,
    scale_x: float,
    scale_y: float,
    physical_size: Tuple[int, int],
) -> Optional[Dict[str, int]]:
    """Convert one frame-local CSS rect to screenshot pixels, or ``None``."""
    try:
        left = math.floor((transform.ox + transform.sx * local["x"]) * scale_x)
        top = math.floor((transform.oy + transform.sy * local["y"]) * scale_y)
        right = math.ceil(
            (transform.ox + transform.sx * (local["x"] + local["width"])) * scale_x
        )
        bottom = math.ceil(
            (transform.oy + transform.sy * (local["y"] + local["height"])) * scale_y
        )
    except (KeyError, TypeError, ValueError):
        return None
    width, height = physical_size
    left = max(0, min(int(left), width))
    top = max(0, min(int(top), height))
    right = max(0, min(int(right), width))
    bottom = max(0, min(int(bottom), height))
    if right - left <= 1 or bottom - top <= 1:
        return None
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _intersect(
    bounds: Dict[str, int], clip: Dict[str, int]
) -> Optional[Dict[str, int]]:
    left = max(bounds["left"], clip["left"])
    top = max(bounds["top"], clip["top"])
    right = min(bounds["right"], clip["right"])
    bottom = min(bounds["bottom"], clip["bottom"])
    if right - left <= 1 or bottom - top <= 1:
        return None
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def clip_tree(
    nodes: Sequence[Dict[str, Any]], clip: Dict[str, int]
) -> List[Dict[str, Any]]:
    """Drop/clip nodes that fall outside an iframe's visible rect."""
    out: List[Dict[str, Any]] = []
    for node in nodes:
        bounds = node.get("boundsInScreen")
        if isinstance(bounds, dict):
            clipped = _intersect(bounds, clip)
            if clipped is None:
                continue
            node = dict(node)
            node["boundsInScreen"] = clipped
        children = clip_tree(node.get("children") or [], clip)
        if not bounds and not children:
            continue
        node = dict(node)
        if children:
            node["children"] = children
        elif "children" in node:
            node["children"] = []
        out.append(node)
    return out


def find_iframe_node(
    nodes: Sequence[Dict[str, Any]], ordinal: int
) -> Optional[Dict[str, Any]]:
    """Find the portal node carrying ``frameOrdinal`` (recursive)."""
    for node in nodes:
        if node.get("frameOrdinal") == ordinal:
            return node
        found = find_iframe_node(node.get("children") or [], ordinal)
        if found is not None:
            return found
    return None


def _frame_order(main_frame: Any) -> List[Any]:
    order = [main_frame]
    queue = [main_frame]
    while queue:
        frame = queue.pop(0)
        for child in list(getattr(frame, "child_frames", ()) or ()):
            order.append(child)
            queue.append(child)
    return order


def _viewport(payload: Dict[str, Any]) -> Tuple[float, float]:
    viewport = payload.get("viewport") or {}
    try:
        width = float(viewport.get("width") or 0)
        height = float(viewport.get("height") or 0)
    except (TypeError, ValueError):
        width, height = 0.0, 0.0
    return width, height


async def _eval_frame(
    frame: Any, options: Dict[str, Any], timeout: float
) -> Optional[Dict[str, Any]]:
    try:
        payload = await asyncio.wait_for(
            frame.evaluate(EXTRACTOR_JS, options), timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001 - one frame must not fail observe
        logger.debug("MobileGym DOM frame extraction skipped: %s", exc)
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("tree"), dict):
        return None
    return payload


async def _frame_placement(frame: Any, timeout: float) -> Optional[FramePlacement]:
    try:
        handle = await asyncio.wait_for(frame.frame_element(), timeout=timeout)
        info = await asyncio.wait_for(
            handle.evaluate(PLACEMENT_JS), timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001 - detached/navigating frame
        logger.debug("MobileGym iframe placement skipped: %s", exc)
        return None
    if not isinstance(info, dict):
        return None
    rect = info.get("rect") or {}
    try:
        placement = FramePlacement(
            ordinal=int(info.get("ordinal", -1)),
            rect=Rect(
                float(rect.get("x", 0)),
                float(rect.get("y", 0)),
                float(rect.get("width", 0)),
                float(rect.get("height", 0)),
            ),
            client_width=float(info.get("client_width", 0)),
            client_height=float(info.get("client_height", 0)),
            border_left=float(info.get("border_left", 0)),
            border_top=float(info.get("border_top", 0)),
            border_right=float(info.get("border_right", 0)),
            border_bottom=float(info.get("border_bottom", 0)),
            padding_left=float(info.get("padding_left", 0)),
            padding_top=float(info.get("padding_top", 0)),
            padding_right=float(info.get("padding_right", 0)),
            padding_bottom=float(info.get("padding_bottom", 0)),
            child_viewport=(0.0, 0.0),
            display=str(info.get("display", "block")),
            visibility=str(info.get("visibility", "visible")),
            opacity=float(info.get("opacity", 1.0)),
            src=str(info.get("src", "")),
            title=str(info.get("title", "")),
        )
    except (TypeError, ValueError):
        return None
    return placement


def _node_class(node: Dict[str, Any]) -> str:
    tag = str(node.get("tag") or "").lower()
    if node.get("editable"):
        return "android.widget.EditText"
    if tag == "select":
        return "android.widget.Spinner"
    if node.get("checkable"):
        if tag == "input" and str(node.get("role") or "").lower() == "radio":
            return "android.widget.RadioButton"
        return "android.widget.CheckBox"
    if tag == "iframe":
        return "android.webkit.WebView"
    if node.get("scrollable"):
        return "android.widget.ScrollView"
    if tag == "img":
        return "android.widget.ImageView"
    if node.get("clickable") or node.get("long_clickable"):
        return "android.widget.Button"
    if node.get("text") or node.get("content_description"):
        return "android.widget.TextView"
    return "android.view.ViewGroup"


def _package_name(phone_state: Dict[str, Any]) -> str:
    return str(phone_state.get("currentApp") or phone_state.get("packageName") or "")


def _materialize(
    node: Dict[str, Any],
    transform: Transform,
    scale_x: float,
    scale_y: float,
    physical_size: Tuple[int, int],
    package: str,
    budget: List[int],
) -> Optional[Dict[str, Any]]:
    children = []
    for child in node.get("children") or []:
        built = _materialize(
            child, transform, scale_x, scale_y, physical_size, package, budget
        )
        if built is not None:
            children.append(built)

    local = node.get("visible_bounds")
    if not isinstance(local, dict):
        if not children:
            return None
        return {
            "className": "android.view.ViewGroup",
            "packageName": package,
            "children": children,
        }

    if budget[0] <= 0:
        return None
    bounds = physical_bounds(local, transform, scale_x, scale_y, physical_size)
    if bounds is None:
        return None
    budget[0] -= 1

    portal: Dict[str, Any] = {
        "boundsInScreen": bounds,
        "className": _node_class(node),
        "text": str(node.get("text") or ""),
        "contentDescription": str(node.get("content_description") or ""),
        "resourceId": str(node.get("resource_id") or ""),
        "packageName": package,
        "hintText": str(node.get("hint_text") or ""),
        "isCheckable": bool(node.get("checkable")),
        "isChecked": bool(node.get("checked")),
        "isClickable": bool(node.get("clickable")),
        "isLongClickable": bool(node.get("long_clickable")),
        "isEditable": bool(node.get("editable")),
        "isFocused": bool(node.get("focused")),
        "isVisibleToUser": bool(node.get("visible_to_user", True)),
        "isScrollable": bool(node.get("scrollable")),
        "scrollDirection": str(node.get("scroll_direction") or "none"),
        "isSelected": bool(node.get("selected")),
        "isEnabled": bool(node.get("enabled", True)),
        # jev-mobile reads ``enabled``; AndroidWorld/MobileWorld consumers read
        # ``isEnabled``. Emit both so one tree serves every policy.
        "enabled": bool(node.get("enabled", True)),
    }
    if str(node.get("tag") or "") == "iframe":
        portal["frameOrdinal"] = int(node.get("frame_ordinal", -1))
        portal["frameUrl"] = str(node.get("frame_url") or "")
        label = str(node.get("title") or "")
        if label and not portal["contentDescription"]:
            portal["contentDescription"] = label
    if children:
        portal["children"] = children
    return portal


def _materialize_tree(
    payload: Dict[str, Any],
    transform: Transform,
    scale_x: float,
    scale_y: float,
    physical_size: Tuple[int, int],
    phone_state: Dict[str, Any],
    budget: List[int],
) -> List[Dict[str, Any]]:
    package = _package_name(phone_state)
    tree = payload.get("tree") or {}
    out: List[Dict[str, Any]] = []
    if tree and not tree.get("structural"):
        # ``build()`` collapses a single-child wrapper, so the root itself can
        # be the only semantic node (e.g. a one-button iframe document).
        built = _materialize(
            tree, transform, scale_x, scale_y, physical_size, package, budget
        )
        if built is not None:
            out.append(built)
        return out
    for node in tree.get("children") or []:
        built = _materialize(
            node, transform, scale_x, scale_y, physical_size, package, budget
        )
        if built is not None:
            out.append(built)
    return out


async def build_dom_tree(
    page: Any,
    *,
    physical_size: Tuple[int, int],
    dpr: Optional[float] = None,
    root_selector: str = "body",
    frame_timeout_s: float = FRAME_TIMEOUT_S,
    total_timeout_s: float = TOTAL_TIMEOUT_S,
    max_total_nodes: int = MAX_TOTAL_NODES,
) -> Dict[str, Any]:
    """Sample the main frame and every iframe, returning a portal state dict.

    Frame failures are isolated; the main frame failing raises so the driver
    can fall back to the empty portal stub rather than failing ``observe``.
    """
    started = time.monotonic()
    deadline = started + max(1.0, float(total_timeout_s))
    main_frame = page.main_frame
    if main_frame is None:
        raise RuntimeError("MobileGym page has no main frame")

    main_payload = await _eval_frame(
        main_frame,
        {
            "rootSelector": root_selector,
            "isMainFrame": True,
            "maxNodes": MAX_NODES_PER_FRAME,
            "maxDepth": MAX_DEPTH,
            "maxScan": MAX_SCAN_ELEMENTS,
            "maxLabelChars": MAX_LABEL_CHARS,
        },
        frame_timeout_s,
    )
    if main_payload is None:
        raise RuntimeError("MobileGym main frame produced no DOM tree")

    viewport = _viewport(main_payload)
    effective_dpr = float(dpr or main_payload.get("dpr") or 3.0)
    scale_x = (
        physical_size[0] / viewport[0] if viewport[0] > 0 else effective_dpr
    )
    scale_y = (
        physical_size[1] / viewport[1] if viewport[1] > 0 else effective_dpr
    )

    frames = _frame_order(main_frame)
    payloads: Dict[Any, Dict[str, Any]] = {main_frame: main_payload}
    transforms: Dict[Any, Transform] = {main_frame: Transform(0.0, 0.0, 1.0, 1.0)}
    roots: Dict[Any, List[Dict[str, Any]]] = {}

    phone_state = dict(main_payload.get("phone_state") or {})
    budget = [max(1, int(max_total_nodes))]
    roots[main_frame] = _materialize_tree(
        main_payload,
        transforms[main_frame],
        scale_x,
        scale_y,
        physical_size,
        phone_state,
        budget,
    )
    truncated = bool(main_payload.get("truncated")) or budget[0] <= 0

    for frame in frames[1:]:
        if len(payloads) >= MAX_FRAMES:
            truncated = True
            break
        if time.monotonic() >= deadline:
            truncated = True
            break
        parent = getattr(frame, "parent_frame", None)
        if parent not in transforms:
            continue
        placement = await _frame_placement(frame, frame_timeout_s)
        if placement is None or not placement.visible():
            continue
        payload = await _eval_frame(
            frame,
            {
                "rootSelector": root_selector,
                "isMainFrame": False,
                "maxNodes": MAX_NODES_PER_FRAME,
                "maxDepth": MAX_DEPTH,
                "maxScan": MAX_SCAN_ELEMENTS,
                "maxLabelChars": MAX_LABEL_CHARS,
            },
            frame_timeout_s,
        )
        if payload is None:
            continue
        viewport_width, viewport_height = _viewport(payload)
        if viewport_width <= 0 or viewport_height <= 0:
            continue
        placement = replace(
            placement, child_viewport=(viewport_width, viewport_height)
        )
        try:
            transforms[frame] = compose_transform(transforms[parent], placement)
        except ValueError:
            continue
        payloads[frame] = payload

        child_roots = _materialize_tree(
            payload,
            transforms[frame],
            scale_x,
            scale_y,
            physical_size,
            phone_state,
            budget,
        )
        if budget[0] <= 0:
            truncated = True
        clipped = clip_tree(
            child_roots, _iframe_physical_bounds(placement, transforms[parent], scale_x, scale_y, physical_size)
        )
        target = find_iframe_node(roots.get(parent, []), placement.ordinal)
        if target is not None and not target.get("isVisibleToUser", True):
            continue
        if target is not None:
            target["className"] = "android.webkit.WebView"
            target["isScrollable"] = False
            if not target.get("contentDescription"):
                target["contentDescription"] = placement.title or placement.src
            if clipped:
                target["children"] = clipped
        elif clipped:
            webview = {
                "boundsInScreen": _iframe_physical_bounds(
                    placement, transforms[parent], scale_x, scale_y, physical_size
                ),
                "className": "android.webkit.WebView",
                "text": "",
                "contentDescription": placement.title or placement.src,
                "resourceId": f"iframe:{placement.ordinal}",
                "packageName": _package_name(phone_state),
                "hintText": "",
                "isCheckable": False,
                "isChecked": False,
                "isClickable": False,
                "isLongClickable": False,
                "isEditable": False,
                "isFocused": False,
                "isVisibleToUser": True,
                "isScrollable": False,
                "scrollDirection": "none",
                "isSelected": False,
                "isEnabled": True,
                "enabled": True,
                "frameOrdinal": placement.ordinal,
                "frameUrl": placement.src,
                "children": clipped,
            }
            roots.setdefault(parent, []).append(webview)
        roots[frame] = clipped

    root = {
        "className": "",
        "text": "",
        "contentDescription": "",
        "resourceId": "",
        "packageName": _package_name(phone_state),
        "boundsInScreen": {
            "left": 0,
            "top": 0,
            "right": int(physical_size[0]),
            "bottom": int(physical_size[1]),
        },
        "children": roots.get(main_frame, []),
    }
    return {
        "a11y_tree": root,
        "device_context": {
            "screen_bounds": {
                "width": int(physical_size[0]),
                "height": int(physical_size[1]),
            },
            "dpr": effective_dpr,
            "viewport": {"width": viewport[0], "height": viewport[1]},
        },
        "phone_state": phone_state,
        "truncated": truncated,
    }


def _iframe_physical_bounds(
    placement: FramePlacement,
    parent: Transform,
    scale_x: float,
    scale_y: float,
    physical_size: Tuple[int, int],
) -> Dict[str, int]:
    bounds = physical_bounds(
        {
            "x": placement.rect.x,
            "y": placement.rect.y,
            "width": placement.rect.width,
            "height": placement.rect.height,
        },
        parent,
        scale_x,
        scale_y,
        physical_size,
    )
    if bounds is None:
        return {
            "left": 0,
            "top": 0,
            "right": int(physical_size[0]),
            "bottom": int(physical_size[1]),
        }
    return bounds


__all__ = [
    "EXTRACTOR_JS",
    "FRAME_TIMEOUT_S",
    "FramePlacement",
    "MAX_FRAMES",
    "MAX_TOTAL_NODES",
    "PLACEMENT_JS",
    "Rect",
    "TOTAL_TIMEOUT_S",
    "Transform",
    "build_dom_tree",
    "clip_tree",
    "compose_transform",
    "find_iframe_node",
    "physical_bounds",
]
