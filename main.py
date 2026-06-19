"""
FastAPI application for handling NetworkX graphs.

This application provides endpoints for:
- Uploading JSON files containing NetworkX graphs
- Retrieving graph summaries by unique ID
- Setting a default graph that can be accessed without uploading a file
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx
import json
import os
import uuid
from typing import Dict, Any
from pathlib import Path

app = FastAPI(
    title="NetworkX Graph API",
    description="API for uploading and analyzing NetworkX graphs",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:5173", "http://127.0.0.1", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded graph files
GRAPH_STORAGE_DIR = "graph_storage"

# Create storage directory if it doesn't exist
Path(GRAPH_STORAGE_DIR).mkdir(exist_ok=True)

# Dictionary to map graph IDs to file paths
# In production, this would be a database
graph_registry: Dict[str, str] = {}

# Default graph ID - special constant to access the default graph
default_graph_id = "default"


def create_degree_centrality_distribution(graph):
    """
    Create a distribution of degree centrality values instead of per-node values.
    This is more compact and suitable for large graphs.
    Uses adaptive binning based on actual degree centrality values.

    Returns:
        A dictionary with bins of degree centrality values and their counts.
    """
    degree_centrality_values = list(nx.degree_centrality(graph).values())

    if not degree_centrality_values:
        return {
            "stats": {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "total_nodes": 0
            }
        }

    # Get min and max values
    min_val = min(degree_centrality_values)
    max_val = max(degree_centrality_values)

    # Use adaptive binning based on actual values
    # Create 10 bins that cover the actual range of values
    if min_val == max_val:
        # All values are the same, just use one bin
        bins = [min_val, min_val]
    else:
        # Create 10 evenly spaced bins covering the actual range
        bins = [min_val + i * (max_val - min_val) / 10 for i in range(11)]

    distribution = {}

    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        bin_key = f"{lower:.4f}-{upper:.4f}"
        count = sum(1 for value in degree_centrality_values if lower <= value < upper)
        distribution[bin_key] = count

    # Add statistics about the distribution
    distribution["stats"] = {
        "min": min_val,
        "max": max_val,
        "mean": sum(degree_centrality_values) / len(degree_centrality_values),
        "median": sorted(degree_centrality_values)[len(degree_centrality_values) // 2],
        "total_nodes": len(degree_centrality_values)
    }

    return distribution


@app.post("/upload/", summary="Upload a NetworkX graph JSON file")
async def upload_graph(file: UploadFile = File(...)):
    """
    Upload a JSON file containing a NetworkX graph.

    The file should contain a NetworkX graph in JSON format (e.g., from nx.readwrite.json_graph.node_link_data).

    Returns:
        A unique ID that can be used to reference this graph in other endpoints.
    """
    # Check if file has .json extension
    if not file.filename.lower().endswith('.json'):
        raise HTTPException(status_code=400, detail="File must have .json extension")

    try:
        # Read the file content
        contents = await file.read()
        data = json.loads(contents)

        # Validate that it's a NetworkX graph by trying to load it
        try:
            graph = nx.node_link_graph(data, edges="links" if "links" in data else "edges")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid NetworkX graph format: {str(e)}"
            )

        # Generate a unique ID for this graph
        graph_id = str(uuid.uuid4())

        # Save the file with the graph ID as filename
        file_path = os.path.join(GRAPH_STORAGE_DIR, f"{graph_id}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f)

        # Register the graph
        graph_registry[graph_id] = file_path

        return JSONResponse(
            status_code=201,
            content={
                "graph_id": graph_id,
                "message": "Graph uploaded successfully",
                "filename": file.filename
            }
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/set-default/{graph_id}", summary="Set a graph as the default graph")
async def set_default_graph(graph_id: str):
    """
    Set a graph as the default graph that can be accessed without uploading a file.

    Args:
        graph_id: The unique ID of the graph to set as default, or the filename in graph_storage (without .json extension).

    Returns:
        Confirmation that the graph is now the default.
    """
    # Check if graph ID exists in registry
    if graph_id in graph_registry:
        # Graph ID found in registry, use it
        file_path = graph_registry[graph_id]
    else:
        # Graph ID not in registry, check if it's a file in graph_storage
        # Try with .json extension first
        file_path = os.path.join(GRAPH_STORAGE_DIR, f"{graph_id}.json")
        if not os.path.exists(file_path):
            # Also try without extension (for backward compatibility)
            file_path = os.path.join(GRAPH_STORAGE_DIR, graph_id)
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Graph ID not found in registry or storage")

        # Add to registry for future reference
        graph_registry[graph_id] = file_path

    # Set this graph as the default
    graph_registry[default_graph_id] = file_path

    return JSONResponse(
        status_code=200,
        content={
            "message": "Graph set as default successfully",
            "default_graph_id": default_graph_id,
            "graph_id": graph_id
        }
    )


@app.get("/summary/{graph_id}", summary="Get graph summary by ID")
async def get_graph_summary(graph_id: str):
    """
    Get a summary of the properties of a NetworkX graph.

    Args:
        graph_id: The unique ID returned when uploading the graph, or "default" to access the default graph.

    Returns:
        A JSON object containing various properties of the graph.
    """
    # Check if graph ID exists in registry
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph ID not found")

    try:
        # Load the graph from file
        file_path = graph_registry[graph_id]
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Load as NetworkX graph
        graph = nx.node_link_graph(data, edges="links" if "links" in data else "edges")

        # Calculate graph properties
        summary = {
            "graph_id": graph_id,
            "basic_properties": {
                "number_of_nodes": graph.number_of_nodes(),
                "number_of_edges": graph.number_of_edges(),
                "is_directed": nx.is_directed(graph),
                "is_weakly_connected": nx.is_weakly_connected(graph) if (nx.is_directed(graph) and graph.number_of_nodes() > 0) else None,
                "is_strongly_connected": nx.is_strongly_connected(graph) if (nx.is_directed(graph) and graph.number_of_nodes() > 0) else None,
                "is_connected": nx.is_connected(graph) if (not nx.is_directed(graph) and graph.number_of_nodes() > 0) else None,
            },
            "degree_properties": {
                "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
                "degree_centrality_distribution": create_degree_centrality_distribution(graph) if graph.number_of_nodes() > 0 else {},
            },
            "node_properties": {
                "node_count": graph.number_of_nodes(),
                "nodes_with_highest_degree": sorted(
                    graph.degree(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]  # Top 5 nodes by degree
            },
            "edge_properties": {
                "edge_count": graph.number_of_edges(),
                "edges_with_highest_weight": sorted(
                    graph.edges(data='weight', default=1),
                    key=lambda x: x[2],
                    reverse=True
                )[:5]  # Top 5 edges by weight (if weighted)
            },
            "edge_type_properties": {
                "edge_type_counts": {
                    edge_type: sum(1 for edge in graph.edges(data=True) if 'Edge Type' in edge[2] and edge[2]['Edge Type'] == edge_type)
                    for edge_type in set(edge[2].get('Edge Type', 'Unknown') for edge in graph.edges(data=True))
                }
            },
            "node_type_properties": {
                "node_type_counts": {
                    node_type: sum(1 for node in graph.nodes(data=True) if 'Node Type' in node[1] and node[1]['Node Type'] == node_type)
                    for node_type in set(node[1].get('Node Type', 'Unknown') for node in graph.nodes(data=True))
                }
            }
        }

        return JSONResponse(content=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing graph: {str(e)}")


@app.get("/node-types/{graph_id}", summary="Get node type counts by graph ID")
async def get_node_type_counts(graph_id: str):
    """
    Get the count of nodes for each type in a NetworkX graph.

    Args:
        graph_id: The unique ID returned when uploading the graph, or "default" to access the default graph.

    Returns:
        A JSON object containing the count of nodes for each type.
    """
    # Check if graph ID exists in registry
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph ID not found")

    try:
        # Load the graph from file
        file_path = graph_registry[graph_id]
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Load as NetworkX graph
        graph = nx.node_link_graph(data, edges="links" if "links" in data else "edges")

        # Count nodes by type
        node_type_counts = {}

        for node in graph.nodes(data=True):
            node_data = node[1]
            if 'Node Type' in node_data:
                node_type = node_data['Node Type']
                node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
            else:
                node_type_counts['Unknown'] = node_type_counts.get('Unknown', 0) + 1

        return JSONResponse(content={
            "graph_id": graph_id,
            "node_type_counts": node_type_counts,
            "total_nodes": graph.number_of_nodes()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing graph: {str(e)}")


@app.get("/edge-types/{graph_id}", summary="Get edge type counts by graph ID")
async def get_edge_type_counts(graph_id: str):
    """
    Get the count of edges for each type in a NetworkX graph.

    Args:
        graph_id: The unique ID returned when uploading the graph, or "default" to access the default graph.

    Returns:
        A JSON object containing the count of edges for each type.
    """
    # Check if graph ID exists in registry
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph ID not found")

    try:
        # Load the graph from file
        file_path = graph_registry[graph_id]
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Load as NetworkX graph
        graph = nx.node_link_graph(data, edges="links" if "links" in data else "edges")

        # Count edges by type
        edge_type_counts = {}

        for edge in graph.edges(data=True):
            edge_data = edge[2]
            if 'Edge Type' in edge_data:
                edge_type = edge_data['Edge Type']
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
            else:
                edge_type_counts['Unknown'] = edge_type_counts.get('Unknown', 0) + 1

        # Capability: does this graph distinguish inferred vs observed edges?
        # (MC3 carries an is_inferred flag; MC1/MC2 do not.) The frontend uses
        # has_inferred to decide whether to show the evidence-type control.
        inferred_count = sum(
            1 for _, _, a in graph.edges(data=True) if 'is_inferred' in a
        )
        n_inferred = sum(
            1 for _, _, a in graph.edges(data=True) if bool(a.get('is_inferred', False))
        )

        return JSONResponse(content={
            "graph_id": graph_id,
            "edge_type_counts": edge_type_counts,
            "total_edges": graph.number_of_edges(),
            "has_inferred": inferred_count > 0,
            "inferred_edges": n_inferred,
            "observed_edges": graph.number_of_edges() - n_inferred,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing graph: {str(e)}")


@app.get('/health/', summary='Health check endpoint to verify the API is running.')
async def health():
    """
    Health check endpoint to verify the API is running.

    Returns:
        Status of the API.
    """
    return JSONResponse(
        content={
            "status": "healthy",
            "graph_count": len(graph_registry)
        }
    )


#  the node-link / ego views :
#  the server computes the slice with NetworkX and returns a small node-link
#  payload, instead of shipping the whole 17k-node graph to the browser.
 
def _sg_load_graph(graph_id):
    """Load a stored graph by id, mirroring the existing endpoints' pattern."""
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph ID not found")
    file_path = graph_registry[graph_id]
    with open(file_path, "r") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="links" if "links" in data else "edges")
 
 
def _sg_label(node_id, attrs):
    """Best-effort human label; falls back to the node id."""
    for key in ("name", "label", "title", "Name", "Label"):
        if key in attrs:
            return str(attrs[key])
    return str(node_id)
 
 
def _sg_serialize(graph, full_degree, wanted_edge_types=None, inferred=None):
    """Serialize a (sub)graph to a compact node-link payload.

    `full_degree` comes from the FULL graph so node sizing reflects true
    importance, not degree within the slice.

    `inferred` optionally filters edges by their `is_inferred` flag (used by
    MC3's evidence-vs-inference question): None = keep all, True = only
    inferred edges, False = only observed edges. Edges without the field are
    treated as observed (is_inferred=False), so datasets that lack it (MC1/MC2)
    are unaffected.
    """
    nodes = [
        {
            "id": str(n),
            "label": _sg_label(n, a),
            "type": a.get("Node Type", "Unknown"),
            "degree": int(full_degree.get(n, 0)),
        }
        for n, a in graph.nodes(data=True)
    ]

    def edge_ok(a):
        if wanted_edge_types is not None and a.get("Edge Type", "Unknown") not in wanted_edge_types:
            return False
        if inferred is not None and bool(a.get("is_inferred", False)) != inferred:
            return False
        return True

    links = [
        {
            "source": str(u),
            "target": str(v),
            "type": a.get("Edge Type", "Unknown"),
            "is_inferred": bool(a.get("is_inferred", False)),
        }
        for u, v, a in graph.edges(data=True)
        if edge_ok(a)
    ]
    return nodes, links
 
 
import re as _re
 
 
def _sg_parse_time(value):
    """(year, month, day) ints from a heterogeneous time value, or None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit() and len(s) <= 4:
        return (int(s), None, None)
    head = s.replace("T", " ").split(" ", 1)[0]
    parts = head.split("-")
    try:
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
            return (int(parts[0]), int(parts[1]), None)
        if parts and parts[0].isdigit():
            return (int(parts[0]), None, None)
    except ValueError:
        pass
    m = _re.search(r"\d{4}", s)
    return (int(m.group()), None, None) if m else None
 
 
def _sg_time_int(parsed, upper=False):
    """Sortable int y*10^4 + m*10^2 + d. Missing parts pad low (lower bound)
    or high (upper bound) so coarse bounds make inclusive windows."""
    if parsed is None:
        return None
    y, m, d = parsed
    m = m if m is not None else (12 if upper else 1)
    d = d if d is not None else (31 if upper else 1)
    return y * 10000 + m * 100 + d
 
 
def _sg_detect_time_field(G):
    """Return ('node'|'edge', field_name) for whichever carries more timestamps."""
    NODE_KEYS = ["release_date", "written_date", "date", "notoriety_date", "year", "timestamp", "time"]
    EDGE_KEYS = ["time", "timestamp", "date", "datetime"]
    nf, ef, nv, ev = None, None, 0, 0
    for k in NODE_KEYS:
        c = sum(1 for _, a in G.nodes(data=True) if a.get(k) not in (None, ""))
        if c > nv:
            nv, nf = c, k
    for k in EDGE_KEYS:
        c = sum(1 for _, _, a in G.edges(data=True) if a.get(k) not in (None, ""))
        if c > ev:
            ev, ef = c, k
    return ("edge", ef) if ev > nv else ("node", nf)
 
 
def _sg_apply_time_window(G, time_from, time_to):
    """Return a subgraph restricted to a time window, keeping the network
    meaningful (see header). If no field/bounds apply, returns G unchanged."""
    if time_from is None and time_to is None:
        return G
    source, field = _sg_detect_time_field(G)
    if not field:
        return G
    lo = _sg_time_int(_sg_parse_time(time_from), upper=False) if time_from else None
    hi = _sg_time_int(_sg_parse_time(time_to), upper=True) if time_to else None
 
    def in_range(v):
        t = _sg_time_int(_sg_parse_time(v))
        return t is not None and (lo is None or t >= lo) and (hi is None or t <= hi)
 
    if source == "edge":
        keep_nodes = set()
        for u, v, a in G.edges(data=True):
            if in_range(a.get(field)):
                keep_nodes.add(u)
                keep_nodes.add(v)
        return G.subgraph(keep_nodes)
    # node-time: in-window nodes plus their neighbours, so people/labels remain
    hits = {n for n, a in G.nodes(data=True) if in_range(a.get(field))}
    keep = set(hits)
    for n in hits:
        keep.update(G.neighbors(n))
    return G.subgraph(keep)
 
 
@app.get("/subgraph/{graph_id}", summary="Get a drawable subgraph (ego or type-filtered)")
async def get_subgraph(
    graph_id: str,
    ego: str | None = None,
    radius: int = 1,
    node_types: str | None = None,
    link_types: str | None = None,
    limit: int = 300,
    time_from: str | None = None,
    time_to: str | None = None,
    inferred: str | None = None,
):
    """Return a small, drawable slice of the graph as node-link JSON.
 
    Modes:
      * ego     ?ego=<id>&radius=<r>
      * filter  ?node_types=A,B&limit=<N>
    Optional ?link_types=X,Y keeps only those edge types.
    Optional ?time_from=&time_to= restricts to a time window first.
    """
    try:
        # Normalise the inferred flag: "true"/"false" -> bool, anything else -> None.
        inf = None
        if inferred is not None:
            iv = str(inferred).strip().lower()
            if iv in ("true", "1", "yes"):
                inf = True
            elif iv in ("false", "0", "no"):
                inf = False
        G = _sg_load_graph(graph_id)
        # Apply the global time window first, then ego/filter selection on top.
        G = _sg_apply_time_window(G, time_from, time_to)
        full_degree = dict(G.degree())
        wanted_edges = (
            None if link_types is None
            else set(t.strip() for t in link_types.split(",") if t.strip())
        )
        truncated = False
 
        if ego is not None:
            if ego not in G:
                match = next((n for n in G.nodes if str(n) == ego), None)
                if match is None:
                    raise HTTPException(status_code=404, detail=f"Node '{ego}' not found")
                ego = match
            H = nx.ego_graph(G, ego, radius=max(1, radius), undirected=True)
            if H.number_of_nodes() > limit:
                keep = sorted(
                    (n for n in H.nodes if n != ego),
                    key=lambda n: full_degree.get(n, 0),
                    reverse=True,
                )[: max(0, limit - 1)]
                keep.append(ego)
                H = G.subgraph(keep)
                truncated = True
            mode = "ego"
        else:
            wanted_nodes = (
                None if node_types is None
                else set(t.strip() for t in node_types.split(",") if t.strip())
            )
            selected = [
                n for n, a in G.nodes(data=True)
                if wanted_nodes is None or a.get("Node Type", "Unknown") in wanted_nodes
            ]
            if len(selected) > limit:
                selected = sorted(selected, key=lambda n: full_degree.get(n, 0), reverse=True)[:limit]
                truncated = True
            H = G.subgraph(selected)
            mode = "filter"
 
        nodes, links = _sg_serialize(H, full_degree, wanted_edges, inferred=inf)
        return JSONResponse(content={
            "graph_id": graph_id,
            "mode": mode,
            "directed": nx.is_directed(G),
            "truncated": truncated,
            "limit": limit,
            "node_count": len(nodes),
            "link_count": len(links),
            "nodes": nodes,
            "links": links,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building subgraph: {str(e)}")
 


## End point for the sankey diagramm  .

@app.get("/type-flows/{graph_id}", summary="Aggregate edge counts by (source group, edge type, target group)")
async def get_type_flows(
    graph_id: str,
    top: int = 0,
    group_by: str = "Node Type",
    focus_source: str | None = None,
    edges: str | None = None,
):
    """
    Type-level 'metagraph' for the Sankey: how many edges connect each
    (source group) -> (Edge Type) -> (target group).

    By default the group is the node's canonical "Node Type", which reproduces
    the original type -> relationship -> type view exactly. Pass
    ?group_by=<attribute> to bucket by ANY other node attribute (e.g. genre)
    instead. This stays domain-agnostic: nothing is hardcoded -- the attribute
    is simply whatever the data carries, discovered via /node-attributes. When
    grouping by a non-type attribute, only edges whose BOTH endpoints carry that
    attribute are counted, so the space stays clean (no catch-all 'Unknown').

    Optional ?focus_source=<value> keeps only flows leaving that group value --
    the drill-down (e.g. focus_source=Oceanus Folk yields Oceanus Folk's outgoing
    influence into other genres). Optional ?edges=A,B restricts to those Edge
    Types. ?top=N keeps the N largest flows. Always computed over the FULL graph.
    """
    try:
        from collections import Counter
        G = _sg_load_graph(graph_id)
        by_type = (group_by == "Node Type")
        want_edges = {s.strip() for s in edges.split(",")} if edges else None
        counter = Counter()
        for u, v, a in G.edges(data=True):
            et = a.get("Edge Type", "Unknown")
            if want_edges is not None and et not in want_edges:
                continue
            sg = G.nodes[u].get(group_by)
            tg = G.nodes[v].get(group_by)
            if by_type:
                sg = sg if sg is not None else "Unknown"
                tg = tg if tg is not None else "Unknown"
            elif sg is None or tg is None:
                # grouping by an arbitrary attribute: skip edges whose endpoints
                # don't both carry it (keeps e.g. the genre space clean).
                continue
            counter[(sg, et, tg)] += 1
        flows = [
            {"source_type": s, "edge_type": e, "target_type": t, "count": c}
            for (s, e, t), c in counter.items()
        ]
        if focus_source is not None:
            flows = [f for f in flows if f["source_type"] == focus_source]
        flows.sort(key=lambda f: f["count"], reverse=True)
        if top and top > 0:
            flows = flows[:top]
        return JSONResponse(content={
            "graph_id": graph_id,
            "group_by": group_by,
            "focus_source": focus_source,
            "flow_count": len(flows),
            "flows": flows,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building type flows: {str(e)}")


@app.get("/node-attributes/{graph_id}", summary="List categorical node attributes usable for grouping")
async def get_node_attributes(graph_id: str):
    """
    Report which node attributes are categorical enough to group or colour by
    (used to populate the Sankey's "group by" selector, and a natural source for
    the future property filters). Domain-agnostic: it just inspects whatever
    attributes the data carries, excluding identity/label and the canonical
    "Node Type". An attribute qualifies if it appears on a meaningful share of
    nodes and has modest cardinality (2..50 distinct values) -- so a music graph
    surfaces `genre`, a maritime graph surfaces its own fields, with no
    per-dataset code.
    """
    try:
        from collections import defaultdict
        G = _sg_load_graph(graph_id)
        total = max(1, G.number_of_nodes())
        coverage = defaultdict(int)
        distinct = defaultdict(set)
        SKIP = {"id", "label", "name", "Node Type"}
        for _, a in G.nodes(data=True):
            for k, val in a.items():
                if k in SKIP or val is None or isinstance(val, (dict, list)):
                    continue
                coverage[k] += 1
                if len(distinct[k]) <= 60:
                    distinct[k].add(str(val))
        groupable = []
        for k, cov in coverage.items():
            nd = len(distinct[k])
            if 2 <= nd <= 50 and cov >= max(2, int(total * 0.02)):
                groupable.append({"key": k, "coverage": cov, "distinct": nd})
        groupable.sort(key=lambda x: -x["coverage"])
        return JSONResponse(content={
            "graph_id": graph_id,
            "groupable": groupable,
            "total_nodes": G.number_of_nodes(),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading node attributes: {str(e)}")



@app.get("/influence-ranking/{graph_id}", summary="Rank nodes most affected by a seed group, optionally rolled up via a relation")
async def get_influence_ranking(
    graph_id: str,
    source_value: str,
    source_attr: str = "Node Type",
    edges: str | None = None,
    direction: str = "incoming",
    via: str | None = None,
    top: int = 15,
):
    """
    Generic "who is most affected by X" ranking. Domain-agnostic: nothing about
    music is hardcoded -- the seed group, the traversed edge types, the traversal
    direction and the roll-up relation are all parameters, so a maritime graph
    could rank, say, vessels most implicated by an event the same way.

    Steps:
      1. Seed S = nodes whose `source_attr` == `source_value`
         (e.g. genre == "Oceanus Folk").
      2. Affected A = nodes linked to S by an edge whose type is in `edges`, in
         the given `direction`. With "incoming", A holds predecessors of S
         (nodes pointing into the seed); with "outgoing", successors. Members of
         S are excluded -- a group is not "affected by" itself. NB MC1 influence
         edges point derivative -> original, so "incoming" = works that derive
         from the seed = works the seed influenced.
      3. If `via` is given, roll each affected node up through that relation
         (e.g. PerformerOf -> performer) and rank the roll-up nodes by the number
         of distinct affected items they account for; otherwise rank A directly
         by how many seed nodes each connects to.

    MC1 example: source_attr=genre, source_value=Oceanus Folk,
    edges=InStyleOf,CoverOf,InterpolatesFrom,DirectlySamples,LyricalReferenceTo,
    direction=incoming, via=PerformerOf -> the artists whose songs most derive
    from Oceanus Folk.
    """
    try:
        from collections import defaultdict
        G = _sg_load_graph(graph_id)
        directed = G.is_directed()
        want_edges = {s.strip() for s in edges.split(",")} if edges else None

        def edge_ok(a):
            return want_edges is None or a.get("Edge Type", "Unknown") in want_edges

        seed = {n for n, a in G.nodes(data=True) if a.get(source_attr) == source_value}

        affected = defaultdict(set)  # affected node -> set of seed nodes it links to
        for u, v, a in G.edges(data=True):
            if not edge_ok(a):
                continue
            if v in seed and u not in seed and (not directed or direction == "incoming"):
                affected[u].add(v)
            if u in seed and v not in seed and (not directed or direction == "outgoing"):
                affected[v].add(u)

        A = set(affected.keys())

        def lbl(n):
            a = G.nodes[n]
            return a.get("name") or a.get("label") or str(n)

        def typ(n):
            return G.nodes[n].get("Node Type", "Unknown")

        if via:
            roll = defaultdict(set)  # roll-up node -> set of affected nodes
            for u, v, a in G.edges(data=True):
                if a.get("Edge Type") != via:
                    continue
                if v in A:
                    roll[u].add(v)
                if u in A:
                    roll[v].add(u)
            ranked = sorted(
                ((n, items) for n, items in roll.items() if n not in A and n not in seed),
                key=lambda kv: len(kv[1]), reverse=True,
            )
            ranking = [{"id": str(n), "label": lbl(n), "type": typ(n), "count": len(items)}
                       for n, items in ranked[:top]]
        else:
            ranked = sorted(affected.items(), key=lambda kv: len(kv[1]), reverse=True)
            ranking = [{"id": str(n), "label": lbl(n), "type": typ(n), "count": len(s)}
                       for n, s in ranked[:top]]

        return JSONResponse(content={
            "graph_id": graph_id,
            "source_attr": source_attr,
            "source_value": source_value,
            "direction": direction,
            "via": via,
            "seed_count": len(seed),
            "affected_count": len(A),
            "ranking": ranking,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building influence ranking: {str(e)}")


@app.get("/search/{graph_id}", summary="Find nodes whose label/id matches a query")
async def search_nodes(graph_id: str, q: str = "", limit: int = 10):
    """
    Case-insensitive substring search over node labels (and ids) for the
    autocomplete. Returns up to `limit` matches as {id, label, type}, which the
    search box shows and whose `id` is passed to /subgraph?ego=<id> for the
    Ego card.
    """
    try:
        query = q.strip().lower()
        if not query:
            return JSONResponse(content={"graph_id": graph_id, "query": q, "matches": []})
 
        G = _sg_load_graph(graph_id)
        matches = []
        for n, a in G.nodes(data=True):
            label = _sg_label(n, a)
            if query in label.lower() or query in str(n).lower():
                matches.append({
                    "id": str(n),
                    "label": label,
                    "type": a.get("Node Type", "Unknown"),
                })
                if len(matches) >= max(1, limit):
                    break
 
        return JSONResponse(content={
            "graph_id": graph_id,
            "query": q,
            "match_count": len(matches),
            "matches": matches,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching nodes: {str(e)}")


#  This endpoint detects the source keys and copies their values into the
#  canonical "Node Type" / "Edge Type" keys the rest of the pipeline expects,
#  so every existing endpoint + the _sg_ helpers work unchanged for any graph.

NODE_TYPE_KEYS = ["Node Type", "type", "node_type", "nodeType", "category", "kind"]
EDGE_TYPE_KEYS = ["Edge Type", "role", "edge_type", "type", "relation", "relationship"]
 
 
def _sg_first_present(attrs, keys):
    for k in keys:
        v = attrs.get(k)
        if v not in (None, ""):
            return v, k
    return None, None
 
 
@app.post("/normalize/{graph_id}", summary="Standardize node/edge type keys for any schema")
async def normalize_graph(graph_id: str):
    """
    Ensure every node has a 'Node Type' and every edge an 'Edge Type', derived
    from whatever key the source graph used, then PERSIST the result to the
    graph's file so every other endpoint reads the standardized keys.
    Idempotent: re-running is harmless. Non-destructive: already-typed records
    (MC1) are left untouched.
    """
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph ID not found")
 
    try:
        file_path = graph_registry[graph_id]
        with open(file_path, "r") as f:
            data = json.load(f)
 
        node_src, edge_src = set(), set()
 
        # --- nodes: work directly on the raw JSON dict (the source of truth) --
        for n in data.get("nodes", []):
            if not n.get("Node Type"):
                val, key = _sg_first_present(n, NODE_TYPE_KEYS)
                if val is not None:
                    n["Node Type"] = val
                    if key:
                        node_src.add(key)
                elif n.get("lat") is not None or n.get("lon") is not None:
                    n["Node Type"] = "place"  # fallback: geo node
                else:
                    n["Node Type"] = "Unknown"
 
        # --- edges: the edge list key is "links" or "edges" depending on file -
        edge_key = "links" if "links" in data else "edges"
        for e in data.get(edge_key, []):
            if not e.get("Edge Type"):
                val, key = _sg_first_present(e, EDGE_TYPE_KEYS)
                if val is not None:
                    e["Edge Type"] = val
                    if key:
                        edge_src.add(key)
                elif e.get("time") is not None:
                    e["Edge Type"] = "travel"  # fallback: timestamped travel leg
                else:
                    e["Edge Type"] = "linked"
 
        # --- persist back so all later reads see the standardized keys --------
        with open(file_path, "w") as f:
            json.dump(data, f)
 
        return JSONResponse(content={
            "graph_id": graph_id,
            "normalized": True,
            "node_type_source_keys": sorted(node_src),
            "edge_type_source_keys": sorted(edge_src),
            "node_count": len(data.get("nodes", [])),
            "edge_count": len(data.get(edge_key, [])),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error normalizing graph: {str(e)}")


## Geo Endpoint 

 
GEO_LAT_KEYS = ["lat", "latitude", "Lat", "Latitude", "y"]
GEO_LON_KEYS = ["lon", "lng", "long", "longitude", "Lon", "Longitude", "x"]
 
 
def _geo_first(attrs, keys):
    for k in keys:
        v = attrs.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                pass
    return None
 
 
@app.get("/geo/{graph_id}", summary="Coordinate-bearing nodes for the spatial map")
async def get_geo(graph_id: str):
    """Return points (and place-to-place links) for any graph that carries
    coordinates. x = longitude-axis, y = latitude-axis (auto-detected)."""
    try:
        G = _sg_load_graph(graph_id)
        full_degree = dict(G.degree())
 
        raw = []  # (node, a, va, vb) where va<-lat-field, vb<-lon-field
        for n, a in G.nodes(data=True):
            va = _geo_first(a, GEO_LAT_KEYS)
            vb = _geo_first(a, GEO_LON_KEYS)
            if va is not None and vb is not None:
                raw.append((n, a, va, vb))
 
        if not raw:
            return JSONResponse(content={
                "graph_id": graph_id, "spatial": False,
                "point_count": 0, "points": [], "links": [],
            })
 
        # Axis detection: whichever field strays outside [-90, 90] is longitude.
        a_vals = [va for _, _, va, _ in raw]
        b_vals = [vb for _, _, _, vb in raw]
        a_is_lat = all(-90 <= v <= 90 for v in a_vals)
        b_is_lat = all(-90 <= v <= 90 for v in b_vals)
        # Default: field A = lat, field B = lon. Flip if the evidence says so.
        if a_is_lat and not b_is_lat:
            lat_pick = lambda va, vb: (va, vb)   # A lat, B lon (normal)
        elif b_is_lat and not a_is_lat:
            lat_pick = lambda va, vb: (vb, va)   # B lat, A lon (swapped)
        else:
            lat_pick = lambda va, vb: (va, vb)   # ambiguous -> trust field names
 
        coord_of = {}
        points = []
        for n, a, va, vb in raw:
            lat, lon = lat_pick(va, vb)           # lat-axis, lon-axis values
            coord_of[n] = True
            points.append({
                "id": str(n),
                "label": _sg_label(n, a),
                "type": a.get("Node Type", a.get("type", "Unknown")),
                "x": lon,                          # longitude-axis  (GeoJSON[0])
                "y": lat,                          # latitude-axis   (GeoJSON[1])
                "zone": a.get("zone") or a.get("zone_detail") or "",
                "degree": int(full_degree.get(n, 0)),
            })
 
        # Links whose BOTH endpoints carry coordinates (drawable as map lines).
        links = []
        for u, v in G.edges():
            if u in coord_of and v in coord_of:
                links.append({"source": str(u), "target": str(v)})
 
        return JSONResponse(content={
            "graph_id": graph_id, "spatial": True,
            "point_count": len(points), "link_count": len(links),
            "points": points, "links": links,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building geo view: {str(e)}")
 

##    timeline endpoint  
##  Adaptive temporal aggregation, with optional category breakdown.  
    

NODE_TIME_KEYS = ["release_date", "written_date", "date", "notoriety_date", "year", "timestamp", "time"]
EDGE_TIME_KEYS = ["time", "timestamp", "date", "datetime"]
_TL_TOP_GROUPS = 6  # max distinct categories before the rest become "Other"
 
 
def _tl_parse(value):
    """Parse a heterogeneous time value into (year, month, day) ints.
 
    Handles year-only strings ('2017'), ISO-ish datetimes ('0040-04-24 21:00:00')
    and zero-padded fictional years. month/day are None when only a year is
    present. Returns None if nothing date-like is found.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit() and len(s) <= 4:                 # year-only, e.g. "2017"
        return (int(s), None, None)
    head = s.replace("T", " ").split(" ", 1)[0]      # date part before any time
    parts = head.split("-")
    try:
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
            return (int(parts[0]), int(parts[1]), None)
        if parts and parts[0].isdigit():
            return (int(parts[0]), None, None)
    except ValueError:
        pass
    import re                                        # last resort: any 4-digit year
    m = re.search(r"\d{4}", s)
    return (int(m.group()), None, None) if m else None
 
 
def _tl_choose_granularity(parsed):
    """Pick 'year' | 'month' | 'day' by readability.
 
    Two guards make it robust:
      1. If the data is essentially year-only (the common case for release years),
         use 'year' even if a few stray records carry a full date. This stops a
         handful of dated rows from shattering a yearly dataset into noisy months.
      2. Otherwise choose the FINEST resolution whose distinct-bucket count stays
         within a readable cap. This is driven by bucket *count*, not raw span, so
         outlier dates or multi-era data can't over-coarsen the axis.
    """
    years, months, days = set(), set(), set()
    md = 0
    for y, m, d in parsed:
        years.add(y)
        if m is not None:
            md += 1
            months.add((y, m))
            days.add((y, m, d or 1))
    # Guard 1: mostly year-only -> year.
    if md < 0.6 * len(parsed):
        return "year"
    # Guard 2: finest readable resolution.
    MAX = 48
    for gran, cnt in (("day", len(days)), ("month", len(months)), ("year", len(years))):
        if 1 <= cnt <= MAX:
            return gran
    return "year"
 
 
def _tl_key(p, gran):
    """Format a bucket key for a parsed value at the chosen granularity."""
    y, m, d = p[0], p[1], p[2]
    if gran == "year" or m is None:
        return f"{y:04d}"
    if gran == "month":
        return f"{y:04d}-{m:02d}"
    return f"{y:04d}-{m:02d}-{(d or 1):02d}"
 
 
@app.get("/timeline/{graph_id}", summary="Adaptive activity-over-time (optionally grouped)")
async def get_timeline(
    graph_id: str,
    ego: str | None = None,
    radius: int = 1,
    bucket: str = "auto",
    group_by: str | None = None,
):
    try:
        from collections import Counter, defaultdict
        G = _sg_load_graph(graph_id)
 
        # Optionally scope to one entity's neighbourhood (same id-resolution as /subgraph).
        scope = "all"
        if ego is not None:
            if ego not in G:
                match = next((n for n in G.nodes if str(n) == ego), None)
                if match is None:
                    raise HTTPException(status_code=404, detail=f"Node '{ego}' not found")
                ego = match
            G = nx.ego_graph(G, ego, radius=max(1, radius), undirected=True)
            scope = "ego"
 
        # Decide whether time lives on nodes or edges: whichever has more
        # timestamped records. This is what makes the view schema-agnostic.
        node_field, edge_field, node_vals, edge_vals = None, None, [], []
        for k in NODE_TIME_KEYS:
            vals = [a[k] for _, a in G.nodes(data=True) if a.get(k) not in (None, "")]
            if len(vals) > len(node_vals):
                node_vals, node_field = vals, k
        for k in EDGE_TIME_KEYS:
            vals = [a[k] for _, _, a in G.edges(data=True) if a.get(k) not in (None, "")]
            if len(vals) > len(edge_vals):
                edge_vals, edge_field = vals, k
        source, field = ("edge", edge_field) if len(edge_vals) > len(node_vals) else ("node", node_field)
 
        # Collect (time_value, group_value) pairs from the chosen dimension.
        records = []
        if source == "edge":
            for _u, _v, a in G.edges(data=True):
                tv = a.get(field)
                if tv in (None, ""):
                    continue
                records.append((tv, a.get(group_by) if group_by else None))
        else:
            for _n, a in G.nodes(data=True):
                tv = a.get(field)
                if tv in (None, ""):
                    continue
                records.append((tv, a.get(group_by) if group_by else None))
 
        parsed = [(p, gv) for tv, gv in records if (p := _tl_parse(tv)) is not None]
        if not parsed:
            return JSONResponse(content={
                "graph_id": graph_id, "scope": scope, "source": source,
                "time_field": field, "granularity": None, "group_by": group_by,
                "groups": [], "total_timestamped": 0, "buckets": [],
            })
 
        gran = bucket if bucket in ("year", "month", "day") else _tl_choose_granularity([p for p, _ in parsed])
 
        # --- ungrouped: simple {key, count} buckets -------------------------
        if not group_by:
            counter = Counter(_tl_key(p, gran) for p, _ in parsed)
            buckets = [{"key": k, "count": c} for k, c in sorted(counter.items())]
            return JSONResponse(content={
                "graph_id": graph_id, "scope": scope, "source": source,
                "time_field": field, "granularity": gran, "group_by": None,
                "groups": [], "total_timestamped": len(parsed), "buckets": buckets,
            })
 
        # --- grouped: keep the top categories, fold the rest into "Other" ---
        group_totals = Counter(str(gv) if gv not in (None, "") else "Unknown" for _, gv in parsed)
        top = [g for g, _ in group_totals.most_common(_TL_TOP_GROUPS)]
        top_set = set(top)
        has_other = any(g not in top_set for g in group_totals)
        groups = top + (["Other"] if has_other else [])
 
        per_bucket = defaultdict(lambda: defaultdict(int))
        for p, gv in parsed:
            g = str(gv) if gv not in (None, "") else "Unknown"
            g = g if g in top_set else "Other"
            per_bucket[_tl_key(p, gran)][g] += 1
 
        buckets = []
        for k in sorted(per_bucket):
            by = per_bucket[k]
            buckets.append({"key": k, "count": sum(by.values()), "by": {g: by.get(g, 0) for g in groups}})
 
        return JSONResponse(content={
            "graph_id": graph_id, "scope": scope, "source": source,
            "time_field": field, "granularity": gran, "group_by": group_by,
            "groups": groups, "total_timestamped": len(parsed), "buckets": buckets,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building timeline: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)