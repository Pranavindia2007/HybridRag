"""
Export a standalone HTML viewer for the fused datasheet knowledge graph.

Usage:
    .venv/bin/python scripts/export_graph_viewer.py
    .venv/bin/python scripts/export_graph_viewer.py --graph data/graph_store/datasheet_fused_graph.gpickle
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GRAPH = PROJECT_ROOT / "data/graph_store/datasheet_fused_graph.gpickle"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/graph_store/datasheet_graph_viewer.html"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _display_name(node_id: Any, attrs: dict[str, Any]) -> str:
    for key in ("name", "label", "title", "id"):
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(node_id)


def _node_group(attrs: dict[str, Any]) -> str:
    return str(attrs.get("__class__") or attrs.get("label") or attrs.get("type") or "Unknown")


def _provenance_pages(attrs: dict[str, Any]) -> list[int]:
    prov = attrs.get("__provenance__")
    if not isinstance(prov, dict):
        return []
    pages = prov.get("pages")
    if not isinstance(pages, list):
        return []
    clean_pages = []
    for page in pages:
        try:
            clean_pages.append(int(page))
        except (TypeError, ValueError):
            continue
    return sorted(set(clean_pages))


def _layout_positions(graph: nx.Graph) -> dict[Any, tuple[float, float]]:
    if not graph:
        return {}
    undirected = graph.to_undirected()
    components = sorted(nx.connected_components(undirected), key=len, reverse=True)
    positions: dict[Any, tuple[float, float]] = {}

    # Layout each component independently, then place components on a loose grid.
    grid_cols = max(1, math.ceil(math.sqrt(len(components))))
    cell_size = 2.8
    for index, component in enumerate(components):
        subgraph = undirected.subgraph(component)
        row, col = divmod(index, grid_cols)
        offset_x = col * cell_size
        offset_y = row * cell_size
        if len(subgraph) == 1:
            node = next(iter(subgraph.nodes))
            positions[node] = (offset_x, offset_y)
            continue
        iterations = 80 if len(subgraph) <= 800 else 45
        scale = 1.0 + min(2.0, math.sqrt(len(subgraph)) / 25)
        sub_positions = nx.spring_layout(subgraph, seed=42, iterations=iterations, scale=scale)
        for node, (x, y) in sub_positions.items():
            positions[node] = (float(x) + offset_x, float(y) + offset_y)
    return positions


def graph_to_payload(graph: nx.Graph) -> dict[str, Any]:
    positions = _layout_positions(graph)
    degree = dict(graph.degree())
    group_counts = Counter(_node_group(attrs) for _, attrs in graph.nodes(data=True))
    edge_label_counts = Counter(str(attrs.get("label") or "edge") for _, _, attrs in graph.edges(data=True))

    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        x, y = positions.get(node_id, (0.0, 0.0))
        group = _node_group(attrs)
        nodes.append({
            "id": str(node_id),
            "name": _display_name(node_id, attrs),
            "group": group,
            "degree": int(degree.get(node_id, 0)),
            "pages": _provenance_pages(attrs),
            "x": x,
            "y": y,
            "attrs": _json_safe(attrs),
        })

    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append({
            "source": str(source),
            "target": str(target),
            "label": str(attrs.get("label") or "edge"),
            "attrs": _json_safe(attrs),
        })

    return {
        "summary": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "directed": graph.is_directed(),
            "groups": dict(group_counts.most_common()),
            "edgeLabels": dict(edge_label_counts.most_common()),
        },
        "nodes": nodes,
        "edges": edges,
    }


def build_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Datasheet Knowledge Graph Viewer</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #1f2528;
      --muted: #667075;
      --line: #d7ddd9;
      --accent: #246b5a;
      --selected: #d94f2b;
      --edge: rgba(74, 88, 92, .34);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111516;
        --panel: #171d1f;
        --text: #edf1ee;
        --muted: #a6b1ae;
        --line: #2b3536;
        --accent: #5dbda5;
        --selected: #ff8a65;
        --edge: rgba(177, 190, 188, .25);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      overflow: hidden;
    }}
    .app {{
      display: grid;
      grid-template-columns: 320px 1fr 360px;
      height: 100vh;
      min-height: 620px;
    }}
    aside {{
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 16px;
      overflow: auto;
    }}
    .details {{
      border-right: 0;
      border-left: 1px solid var(--line);
    }}
    h1, h2 {{
      margin: 0 0 10px;
      font-size: 16px;
      line-height: 1.2;
    }}
    h2 {{ margin-top: 18px; font-size: 13px; text-transform: uppercase; color: var(--muted); }}
    .stats {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0 16px;
    }}
    .stat {{
      border: 1px solid var(--line);
      padding: 8px;
      background: color-mix(in srgb, var(--panel) 88%, var(--bg));
    }}
    .stat strong {{ display: block; font-size: 18px; }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      background: var(--bg);
      color: var(--text);
      padding: 8px;
      font: inherit;
      margin: 4px 0 10px;
    }}
    label {{ display: block; color: var(--muted); font-size: 12px; }}
    button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }}
    .toolbar {{
      position: absolute;
      left: 336px;
      top: 12px;
      z-index: 2;
      display: flex;
      gap: 8px;
    }}
    main {{ position: relative; min-width: 0; }}
    canvas {{ width: 100%; height: 100%; display: block; }}
    .legend-row, .node-row {{
      display: grid;
      grid-template-columns: 14px 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 5px 0;
      border-bottom: 1px solid color-mix(in srgb, var(--line) 55%, transparent);
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }}
    .tiny {{ color: var(--muted); font-size: 12px; }}
    .node-row {{
      grid-template-columns: 1fr auto;
      cursor: pointer;
    }}
    .node-row:hover {{ color: var(--accent); }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      border: 1px solid var(--line);
      background: var(--bg);
      padding: 10px;
      max-height: 42vh;
      overflow: auto;
    }}
    .pill {{
      display: inline-block;
      border: 1px solid var(--line);
      padding: 2px 6px;
      margin: 2px 4px 2px 0;
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Datasheet Knowledge Graph</h1>
      <div class="stats">
        <div class="stat"><strong id="nodeCount"></strong><span class="tiny">nodes</span></div>
        <div class="stat"><strong id="edgeCount"></strong><span class="tiny">edges</span></div>
      </div>
      <label for="search">Search node names and attributes</label>
      <input id="search" type="search" placeholder="Package, ENDINIT, ADC, register...">
      <label for="groupFilter">Node type</label>
      <select id="groupFilter"></select>
      <label for="degreeFilter">Minimum degree: <span id="degreeValue">0</span></label>
      <input id="degreeFilter" type="range" min="0" max="20" value="0">
      <button id="resetView" type="button">Reset View</button>
      <h2>Types</h2>
      <div id="legend"></div>
      <h2>Matching Nodes</h2>
      <div id="matches"></div>
    </aside>
    <main>
      <div class="toolbar">
        <button id="zoomIn" type="button">+</button>
        <button id="zoomOut" type="button">-</button>
        <button id="fitView" type="button">Fit</button>
      </div>
      <canvas id="graph"></canvas>
    </main>
    <aside class="details">
      <h1 id="detailTitle">Select a node</h1>
      <div id="detailMeta" class="tiny"></div>
      <h2>Pages</h2>
      <div id="detailPages" class="tiny">No node selected.</div>
      <h2>Neighbors</h2>
      <div id="detailNeighbors" class="tiny">No node selected.</div>
      <h2>Attributes</h2>
      <pre id="detailAttrs">{{}}</pre>
    </aside>
  </div>
  <script>
    const payload = {payload_json};
    const nodes = payload.nodes;
    const edges = payload.edges;
    const byId = new Map(nodes.map(n => [n.id, n]));
    const neighbors = new Map(nodes.map(n => [n.id, []]));
    edges.forEach(e => {{
      neighbors.get(e.source)?.push({{ id: e.target, label: e.label }});
      neighbors.get(e.target)?.push({{ id: e.source, label: e.label }});
    }});

    const colors = ["#246b5a","#8f5c2d","#4b6ea8","#9b4d70","#6f7a2c","#6a5aa8","#2f7f8f","#ad5f45","#61717a","#3e8a58"];
    const groups = [...new Set(nodes.map(n => n.group))].sort();
    const colorFor = new Map(groups.map((g, i) => [g, colors[i % colors.length]]));
    function cssVar(name) {{
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }}

    const state = {{
      scale: 1,
      offsetX: 0,
      offsetY: 0,
      selected: null,
      filtered: new Set(nodes.map(n => n.id)),
      dragging: false,
      lastX: 0,
      lastY: 0
    }};

    const canvas = document.getElementById("graph");
    const ctx = canvas.getContext("2d");
    const search = document.getElementById("search");
    const groupFilter = document.getElementById("groupFilter");
    const degreeFilter = document.getElementById("degreeFilter");
    const degreeValue = document.getElementById("degreeValue");

    document.getElementById("nodeCount").textContent = payload.summary.nodes.toLocaleString();
    document.getElementById("edgeCount").textContent = payload.summary.edges.toLocaleString();

    groupFilter.innerHTML = '<option value="">All types</option>' + groups.map(g => `<option value="${{escapeHtml(g)}}">${{escapeHtml(g)}} (${{payload.summary.groups[g] || 0}})</option>`).join("");
    document.getElementById("legend").innerHTML = groups.map(g => `
      <div class="legend-row">
        <span class="swatch" style="background:${{colorFor.get(g)}}"></span>
        <span>${{escapeHtml(g)}}</span>
        <span class="tiny">${{payload.summary.groups[g] || 0}}</span>
      </div>`).join("");

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}

    function searchableText(node) {{
      return JSON.stringify([node.id, node.name, node.group, node.attrs]).toLowerCase();
    }}

    function applyFilters() {{
      const q = search.value.trim().toLowerCase();
      const group = groupFilter.value;
      const minDegree = Number(degreeFilter.value);
      degreeValue.textContent = String(minDegree);
      state.filtered = new Set(nodes.filter(n => {{
        return (!q || searchableText(n).includes(q)) &&
          (!group || n.group === group) &&
          n.degree >= minDegree;
      }}).map(n => n.id));
      updateMatches();
      draw();
    }}

    function updateMatches() {{
      const matches = nodes.filter(n => state.filtered.has(n.id)).sort((a,b) => b.degree - a.degree).slice(0, 40);
      document.getElementById("matches").innerHTML = matches.map(n => `
        <div class="node-row" data-id="${{escapeHtml(n.id)}}">
          <span>${{escapeHtml(n.name)}}</span>
          <span class="tiny">${{escapeHtml(n.group)}} · ${{n.degree}}</span>
        </div>`).join("") || '<div class="tiny">No matches.</div>';
      document.querySelectorAll(".node-row").forEach(row => {{
        row.addEventListener("click", () => selectNode(row.dataset.id));
      }});
    }}

    function resize() {{
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
      canvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
      draw();
    }}

    function graphToScreen(x, y) {{
      return {{
        x: x * state.scale + state.offsetX,
        y: y * state.scale + state.offsetY
      }};
    }}

    function screenToGraph(x, y) {{
      return {{
        x: (x - state.offsetX) / state.scale,
        y: (y - state.offsetY) / state.scale
      }};
    }}

    function fit() {{
      const rect = canvas.getBoundingClientRect();
      const visible = nodes.filter(n => state.filtered.has(n.id));
      const xs = visible.map(n => n.x);
      const ys = visible.map(n => n.y);
      if (!xs.length) return;
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const pad = 48;
      const scaleX = (rect.width - pad * 2) / Math.max(.001, maxX - minX);
      const scaleY = (rect.height - pad * 2) / Math.max(.001, maxY - minY);
      state.scale = Math.max(18, Math.min(180, Math.min(scaleX, scaleY)));
      state.offsetX = rect.width / 2 - ((minX + maxX) / 2) * state.scale;
      state.offsetY = rect.height / 2 - ((minY + maxY) / 2) * state.scale;
      draw();
    }}

    function draw() {{
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      const selectedNeighbors = new Set(state.selected ? neighbors.get(state.selected)?.map(n => n.id) || [] : []);

      ctx.lineWidth = 1;
      edges.forEach(e => {{
        if (!state.filtered.has(e.source) || !state.filtered.has(e.target)) return;
        const s = byId.get(e.source), t = byId.get(e.target);
        const a = graphToScreen(s.x, s.y), b = graphToScreen(t.x, t.y);
        const related = state.selected && (e.source === state.selected || e.target === state.selected);
        ctx.strokeStyle = related ? cssVar("--selected") : cssVar("--edge");
        ctx.globalAlpha = state.selected && !related ? .16 : .75;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }});
      ctx.globalAlpha = 1;

      nodes.forEach(n => {{
        if (!state.filtered.has(n.id)) return;
        const p = graphToScreen(n.x, n.y);
        const selected = n.id === state.selected;
        const related = selectedNeighbors.has(n.id);
        const r = selected ? 8 : Math.max(3, Math.min(7, 3 + Math.sqrt(n.degree) * .55));
        ctx.fillStyle = selected ? cssVar("--selected") : colorFor.get(n.group);
        ctx.globalAlpha = state.selected && !selected && !related ? .32 : .95;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fill();
        if (selected || (related && state.scale > 40)) {{
          ctx.globalAlpha = 1;
          ctx.fillStyle = cssVar("--text");
          ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
          ctx.fillText(n.name.slice(0, 48), p.x + r + 4, p.y - r - 2);
        }}
      }});
      ctx.globalAlpha = 1;
    }}

    function nearestNode(x, y) {{
      const g = screenToGraph(x, y);
      let best = null;
      let bestDist = Infinity;
      nodes.forEach(n => {{
        if (!state.filtered.has(n.id)) return;
        const dx = n.x - g.x;
        const dy = n.y - g.y;
        const d = dx * dx + dy * dy;
        if (d < bestDist) {{
          best = n;
          bestDist = d;
        }}
      }});
      const threshold = Math.pow(16 / state.scale, 2);
      return bestDist <= threshold ? best : null;
    }}

    function selectNode(id) {{
      state.selected = id;
      const n = byId.get(id);
      if (!n) return;
      document.getElementById("detailTitle").textContent = n.name;
      document.getElementById("detailMeta").textContent = `${{n.group}} · degree ${{n.degree}} · ${{n.id}}`;
      document.getElementById("detailPages").innerHTML = n.pages.length ? n.pages.map(p => `<span class="pill">p. ${{p}}</span>`).join("") : "No page provenance.";
      const ns = neighbors.get(id) || [];
      document.getElementById("detailNeighbors").innerHTML = ns.slice(0, 80).map(edge => {{
        const other = byId.get(edge.id);
        return `<div class="node-row" data-id="${{escapeHtml(edge.id)}}"><span>${{escapeHtml(other?.name || edge.id)}}</span><span class="tiny">${{escapeHtml(edge.label)}}</span></div>`;
      }}).join("") || "No neighbors.";
      document.getElementById("detailAttrs").textContent = JSON.stringify(n.attrs, null, 2);
      document.querySelectorAll(".details .node-row").forEach(row => row.addEventListener("click", () => selectNode(row.dataset.id)));
      draw();
    }}

    canvas.addEventListener("click", e => {{
      const rect = canvas.getBoundingClientRect();
      const node = nearestNode(e.clientX - rect.left, e.clientY - rect.top);
      if (node) selectNode(node.id);
    }});
    canvas.addEventListener("mousedown", e => {{
      state.dragging = true;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
    }});
    window.addEventListener("mousemove", e => {{
      if (!state.dragging) return;
      state.offsetX += e.clientX - state.lastX;
      state.offsetY += e.clientY - state.lastY;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      draw();
    }});
    window.addEventListener("mouseup", () => state.dragging = false);
    canvas.addEventListener("wheel", e => {{
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const before = screenToGraph(e.clientX - rect.left, e.clientY - rect.top);
      const factor = e.deltaY < 0 ? 1.12 : .89;
      state.scale = Math.max(8, Math.min(420, state.scale * factor));
      const after = graphToScreen(before.x, before.y);
      state.offsetX += (e.clientX - rect.left) - after.x;
      state.offsetY += (e.clientY - rect.top) - after.y;
      draw();
    }}, {{ passive: false }});

    search.addEventListener("input", applyFilters);
    groupFilter.addEventListener("change", applyFilters);
    degreeFilter.addEventListener("input", applyFilters);
    document.getElementById("zoomIn").addEventListener("click", () => {{ state.scale *= 1.2; draw(); }});
    document.getElementById("zoomOut").addEventListener("click", () => {{ state.scale /= 1.2; draw(); }});
    document.getElementById("fitView").addEventListener("click", fit);
    document.getElementById("resetView").addEventListener("click", () => {{
      search.value = "";
      groupFilter.value = "";
      degreeFilter.value = "0";
      state.selected = null;
      applyFilters();
      fit();
    }});
    window.addEventListener("resize", resize);

    applyFilters();
    resize();
    fit();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an interactive HTML viewer for the fused graph.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="Input .gpickle graph path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output HTML path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.graph.exists():
        raise FileNotFoundError(f"Graph file not found: {args.graph}. Run ingest.py first.")

    with args.graph.open("rb") as f:
        graph = pickle.load(f)

    payload = graph_to_payload(graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote graph viewer: {args.output.resolve()}")
    print(f"Graph: {payload['summary']['nodes']} nodes, {payload['summary']['edges']} edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
