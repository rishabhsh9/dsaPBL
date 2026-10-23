"""
graph.py - Core Delivery Graph Representation for Smart Logistics System
Models a city delivery network with warehouse and delivery points as a
weighted undirected graph using adjacency lists.
"""

import math
from typing import Dict, List, Tuple, Optional


class DeliveryGraph:
    """
    Weighted undirected graph representing a delivery network.
    Uses adjacency lists for efficient sparse graph traversal O(V + E).
    Each node has 2D spatial coordinates (x, y) representing geographic locations.
    """

    def __init__(self):
        # Adjacency list: node_id -> list of (neighbor_id, distance_km)
        self.adj: Dict[str, List[Tuple[str, float]]] = {}
        # Node metadata: node_id -> {"name": str, "x": float, "y": float, "is_warehouse": bool}
        self.nodes: Dict[str, dict] = {}

    def add_node(self, node_id: str, name: str, x: float, y: float, is_warehouse: bool = False):
        """Add a location node with coordinates to the graph."""
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "name": name,
                "x": float(x),
                "y": float(y),
                "is_warehouse": is_warehouse,
            }
            self.adj[node_id] = []

    def add_edge(self, u: str, v: str, weight: float, bidirectional: bool = True):
        """
        Add a weighted road between node u and node v.
        Weight represents road distance in kilometers.
        """
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Nodes {u} and {v} must be added to graph before adding an edge.")

        # Ensure weight is at least Euclidean distance to guarantee A* admissibility
        eucl = self.euclidean_distance(u, v)
        if weight < round(eucl, 2):
            # Enforce physical reality: road distance >= straight-line displacement
            weight = round(eucl * 1.05, 2)

        self.adj[u].append((v, float(weight)))
        if bidirectional:
            self.adj[v].append((u, float(weight)))

    def get_neighbors(self, u: str) -> List[Tuple[str, float]]:
        """Return list of (neighbor, weight) for a given node."""
        return self.adj.get(u, [])

    def get_nodes(self) -> List[str]:
        """Return list of all node IDs."""
        return list(self.nodes.keys())

    def get_coordinates(self, u: str) -> Tuple[float, float]:
        """Return (x, y) coordinates of node u."""
        node = self.nodes[u]
        return node["x"], node["y"]

    def euclidean_distance(self, u: str, v: str) -> float:
        """
        Compute straight-line Euclidean distance between node u and node v.
        Used as the admissible heuristic h(n) in A* search.
        h(u, v) = sqrt((x_u - x_v)^2 + (y_u - y_v)^2)
        """
        x1, y1 = self.get_coordinates(u)
        x2, y2 = self.get_coordinates(v)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        # Each undirected edge appears twice in adjacency list
        total_directed = sum(len(neighbors) for neighbors in self.adj.values())
        return total_directed // 2


def create_delivery_network() -> DeliveryGraph:
    """
    Creates and populates the logistics network specification:
    - 1 Central Warehouse (WH)
    - 14 Delivery Points (D1 to D14)
    - 26 Interconnecting Roads with realistic kilometer distances
    """
    graph = DeliveryGraph()

    # --- Node Definitions (1 Warehouse + 14 Delivery Destinations) ---
    # Coordinates in km relative to city center (50, 50)
    locations = [
        ("WH",  "Central Logistics Hub",      50.0, 50.0, True),
        ("D1",  "Downtown Financial Center",  48.0, 62.0, False),
        ("D2",  "North Industrial Park",      38.0, 78.0, False),
        ("D3",  "Northeast Tech Corridor",    68.0, 82.0, False),
        ("D4",  "East River Residential",     76.0, 60.0, False),
        ("D5",  "Harbor Logistics Port",      90.0, 48.0, False),
        ("D6",  "Southeast Retail Hub",       82.0, 30.0, False),
        ("D7",  "South Community Center",     58.0, 18.0, False),
        ("D8",  "Southwest Suburban District",36.0, 16.0, False),
        ("D9",  "West Medical Campus",        20.0, 35.0, False),
        ("D10", "West Innovation Park",       18.0, 54.0, False),
        ("D11", "Northwest Commercial Plaza", 28.0, 72.0, False),
        ("D12", "East Midtown Station",       64.0, 52.0, False),
        ("D13", "West Midtown Station",       40.0, 44.0, False),
        ("D14", "International Airport Cargo",72.0, 74.0, False),
    ]

    for node_id, name, x, y, is_wh in locations:
        graph.add_node(node_id, name, x, y, is_warehouse=is_wh)

    # --- 26 Interconnecting Roads (Undirected Edges with distance in km) ---
    # Weights satisfy weight >= Euclidean distance for admissible A* heuristic
    roads = [
        # Arterials connecting Warehouse to key transit hubs
        ("WH",  "D1",  14.5),
        ("WH",  "D12", 15.0),
        ("WH",  "D13", 12.0),
        ("WH",  "D7",  33.5),

        # Northern sector connections
        ("D1",  "D11", 23.0),
        ("D1",  "D2",  19.5),
        ("D1",  "D14", 27.5),
        ("D2",  "D11", 12.0),
        ("D2",  "D3",  31.0),
        ("D11", "D10", 21.0),

        # North-East and Airport connections
        ("D3",  "D14", 10.5),
        ("D14", "D4",  16.0),
        ("D3",  "D4",  24.0),

        # Eastern & Harbor sector connections
        ("D12", "D4",  15.0),
        ("D12", "D5",  27.0),
        ("D4",  "D5",  18.5),
        ("D5",  "D6",  20.0),

        # Southern sector connections
        ("D12", "D6",  29.0),
        ("D6",  "D7",  27.0),
        ("D7",  "D8",  22.5),

        # Western sector connections
        ("D13", "D8",  29.0),
        ("D13", "D9",  22.0),
        ("D13", "D10", 24.5),
        ("D8",  "D9",  25.0),
        ("D9",  "D10", 19.5),

        # East-West bypass connector
        ("D13", "D12", 25.5),
    ]

    for u, v, w in roads:
        graph.add_edge(u, v, w)

    return graph


if __name__ == "__main__":
    net = create_delivery_network()
    print(f"Network loaded: {net.num_nodes()} nodes (1 Warehouse + 14 Deliveries), {net.num_edges()} roads.")
