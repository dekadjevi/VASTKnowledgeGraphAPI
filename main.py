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

        return JSONResponse(content={
            "graph_id": graph_id,
            "edge_type_counts": edge_type_counts,
            "total_edges": graph.number_of_edges()
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
 
 
def _sg_serialize(graph, full_degree, wanted_edge_types=None):
    """Serialize a (sub)graph to a compact node-link payload.
 
    `full_degree` comes from the FULL graph so node sizing reflects true
    importance, not degree within the slice.
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
    links = [
        {"source": str(u), "target": str(v), "type": a.get("Edge Type", "Unknown")}
        for u, v, a in graph.edges(data=True)
        if wanted_edge_types is None or a.get("Edge Type", "Unknown") in wanted_edge_types
    ]
    return nodes, links
 
 
@app.get("/subgraph/{graph_id}", summary="Get a drawable subgraph (ego or type-filtered)")
async def get_subgraph(
    graph_id: str,
    ego: str | None = None,
    radius: int = 1,
    node_types: str | None = None,
    link_types: str | None = None,
    limit: int = 300,
):
    """
    Return a small, drawable slice of the graph as node-link JSON.
 
    Two modes:
      * ego mode     ?ego=<node_id>&radius=<r>      neighborhood around a node
      * filter mode  ?node_types=A,B&limit=<N>      induced subgraph of those
                      node types, capped to the top-N nodes by full-graph degree
 
    Optional ?link_types=X,Y keeps only edges of those types in the result.
    """
    try:
        G = _sg_load_graph(graph_id)
        full_degree = dict(G.degree())
        # Distinguish "param absent" (None -> no filter) from "param present but
        # empty" (-> select nothing). An empty string must NOT mean "all".
        wanted_edges = (
            None if link_types is None
            else set(t.strip() for t in link_types.split(",") if t.strip())
        )
        truncated = False
 
        if ego is not None:
            if ego not in G:
                # node ids may be ints while the query param arrives as a string
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
 
        nodes, links = _sg_serialize(H, full_degree, wanted_edges)
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


## End point for the sankey diagramm . 

@app.get("/type-flows/{graph_id}", summary="Aggregate edge counts by (source type, edge type, target type)")
async def get_type_flows(graph_id: str, top: int = 0):
    """
    Return the type-level 'metagraph' for a Sankey: how many edges connect each
    (source Node Type) -> (Edge Type) -> (target Node Type). Optional ?top=N
    keeps only the N largest flows (recommended for readability). Always
    computed over the FULL graph.
    """
    try:
        from collections import Counter
        G = _sg_load_graph(graph_id)
        counter = Counter()
        for u, v, a in G.edges(data=True):
            st = G.nodes[u].get("Node Type", "Unknown")
            tt = G.nodes[v].get("Node Type", "Unknown")
            et = a.get("Edge Type", "Unknown")
            counter[(st, et, tt)] += 1
        flows = [
            {"source_type": s, "edge_type": e, "target_type": t, "count": c}
            for (s, e, t), c in counter.items()
        ]
        flows.sort(key=lambda f: f["count"], reverse=True)
        if top and top > 0:
            flows = flows[:top]
        return JSONResponse(content={
            "graph_id": graph_id,
            "flow_count": len(flows),
            "flows": flows,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building type flows: {str(e)}")



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
 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)