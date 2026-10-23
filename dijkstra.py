"""
dijkstra.py - Dijkstra's Shortest Path Algorithm
Implements Dijkstra's algorithm using a min-priority queue (heapq).
Used for:
  1. Single-destination routing from Warehouse to any Delivery Point.
  2. All-pairs shortest path computation to build the TSP distance matrix.

Time Complexity: O((V + E) log V)
Space Complexity: O(V)
"""

import heapq
from typing import Dict, List, Tuple, Optional
from graph import DeliveryGraph


class DijkstraResult:
    """Encapsulates the output of a Dijkstra search query."""

    def __init__(self,
                 start: str,
                 target: Optional[str],
                 path: List[str],
                 cost: float,
                 nodes_explored: int,
                 exploration_order: List[str],
                 distances: Dict[str, float],
                 parents: Dict[str, Optional[str]]):
        self.start = start
        self.target = target
        self.path = path
        self.cost = cost
        self.nodes_explored = nodes_explored
        self.exploration_order = exploration_order
        self.distances = distances
        self.parents = parents

    def __repr__(self) -> str:
        path_str = " -> ".join(self.path) if self.path else "None"
        return f"DijkstraResult(cost={self.cost:.2f} km, nodes_explored={self.nodes_explored}, path={path_str})"


def dijkstra(graph: DeliveryGraph, start: str, target: Optional[str] = None) -> DijkstraResult:
    """
    Computes shortest path from start node to target (or all nodes if target is None).

    Parameters:
    - graph: DeliveryGraph instance with adjacency list
    - start: Node ID of starting location (e.g., 'WH')
    - target: Node ID of destination location (e.g., 'D5'), or None for all destinations

    Returns:
    - DijkstraResult containing shortest path, cost, nodes explored, and distance dict.
    """
    if start not in graph.nodes:
        raise ValueError(f"Start node '{start}' does not exist in graph.")
    if target is not None and target not in graph.nodes:
        raise ValueError(f"Target node '{target}' does not exist in graph.")

    # Distance table: node -> shortest known distance from start
    distances: Dict[str, float] = {node: float("inf") for node in graph.get_nodes()}
    distances[start] = 0.0

    # Parent pointers for reconstructing the optimal path
    parents: Dict[str, Optional[str]] = {node: None for node in graph.get_nodes()}

    # Min-Heap Priority Queue storing (cumulative_distance, node_id)
    pq: List[Tuple[float, str]] = [(0.0, start)]

    # Set of visited/settled nodes
    visited = set()

    # Metrics for analysis
    nodes_explored = 0
    exploration_order: List[str] = []

    while pq:
        curr_dist, u = heapq.heappop(pq)

        # If already finalized with a shorter path, skip (lazy deletion handling)
        if u in visited:
            continue

        visited.add(u)
        nodes_explored += 1
        exploration_order.append(u)

        # Early termination: if single-destination query and target is reached
        if target is not None and u == target:
            break

        # Relax all outgoing edges from u
        for v, weight in graph.get_neighbors(u):
            if v in visited:
                continue

            new_dist = curr_dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                parents[v] = u
                heapq.heappush(pq, (new_dist, v))

    # Path reconstruction from target back to start
    path: List[str] = []
    total_cost = float("inf")

    if target is not None:
        if distances[target] < float("inf"):
            curr: Optional[str] = target
            while curr is not None:
                path.append(curr)
                curr = parents[curr]
            path.reverse()
            total_cost = distances[target]
    else:
        # When target is None, path is not defined for a single destination
        total_cost = 0.0

    return DijkstraResult(
        start=start,
        target=target,
        path=path,
        cost=total_cost,
        nodes_explored=nodes_explored,
        exploration_order=exploration_order,
        distances=distances,
        parents=parents,
    )
