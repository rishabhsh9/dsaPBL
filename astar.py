"""
astar.py - A* Search Algorithm with Euclidean Distance Heuristic
Implements A* heuristic search to optimize single-destination logistics delivery.
Uses Euclidean (straight-line) distance as an admissible and consistent heuristic function.

Key Properties:
  - Admissible: h(u) <= true_distance(u, target) because road distance >= straight-line.
  - Consistent: h(u) <= c(u, v) + h(v) satisfies the triangle inequality.
  - Guarantees exact shortest path identical to Dijkstra while expanding fewer nodes.

Time Complexity: O((V + E) log V) worst-case, significantly faster in practice.
Space Complexity: O(V)
"""

import heapq
from typing import Dict, List, Tuple, Optional
from graph import DeliveryGraph


class AStarResult:
    """Encapsulates the output of an A* search query."""

    def __init__(self,
                 start: str,
                 target: str,
                 path: List[str],
                 cost: float,
                 nodes_explored: int,
                 exploration_order: List[str],
                 g_scores: Dict[str, float],
                 heuristic_vals: Dict[str, float]):
        self.start = start
        self.target = target
        self.path = path
        self.cost = cost
        self.nodes_explored = nodes_explored
        self.exploration_order = exploration_order
        self.g_scores = g_scores
        self.heuristic_vals = heuristic_vals

    def __repr__(self) -> str:
        path_str = " -> ".join(self.path) if self.path else "None"
        return f"AStarResult(cost={self.cost:.2f} km, nodes_explored={self.nodes_explored}, path={path_str})"


def astar(graph: DeliveryGraph, start: str, target: str) -> AStarResult:
    """
    Computes shortest path from start node to target using A* with Euclidean heuristic.

    Parameters:
    - graph: DeliveryGraph instance with coordinates
    - start: Node ID of starting location (e.g., 'WH')
    - target: Node ID of destination location (e.g., 'D5')

    Returns:
    - AStarResult containing optimal path, cost, nodes explored, and heuristic data.
    """
    if start not in graph.nodes:
        raise ValueError(f"Start node '{start}' does not exist in graph.")
    if target not in graph.nodes:
        raise ValueError(f"Target node '{target}' does not exist in graph.")

    # g_score[u]: exact shortest distance from start to u found so far
    g_score: Dict[str, float] = {node: float("inf") for node in graph.get_nodes()}
    g_score[start] = 0.0

    # Cache heuristic values h(u) = euclidean_distance(u, target)
    h_cache: Dict[str, float] = {}
    for node in graph.get_nodes():
        h_cache[node] = graph.euclidean_distance(node, target)

    # f_score[u] = g_score[u] + h(u)
    f_score: Dict[str, float] = {node: float("inf") for node in graph.get_nodes()}
    f_score[start] = h_cache[start]

    # Parent pointers for path reconstruction
    parents: Dict[str, Optional[str]] = {node: None for node in graph.get_nodes()}

    # Min-Heap Priority Queue storing: (f_score, g_score, node_id)
    # Tie-breaking by g_score prioritizes nodes closer to goal
    pq: List[Tuple[float, float, str]] = [(f_score[start], 0.0, start)]

    # Set of visited / finalized nodes
    visited = set()

    # Search tracking metrics
    nodes_explored = 0
    exploration_order: List[str] = []

    while pq:
        curr_f, curr_g, u = heapq.heappop(pq)

        if u in visited:
            continue

        visited.add(u)
        nodes_explored += 1
        exploration_order.append(u)

        # Goal test: reached destination
        if u == target:
            break

        # Explore neighbors
        for v, weight in graph.get_neighbors(u):
            if v in visited:
                continue

            tentative_g = curr_g + weight

            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                parents[v] = u
                f_val = tentative_g + h_cache[v]
                f_score[v] = f_val
                heapq.heappush(pq, (f_val, tentative_g, v))

    # Path reconstruction
    path: List[str] = []
    total_cost = float("inf")

    if g_score[target] < float("inf"):
        curr: Optional[str] = target
        while curr is not None:
            path.append(curr)
            curr = parents[curr]
        path.reverse()
        total_cost = g_score[target]

    return AStarResult(
        start=start,
        target=target,
        path=path,
        cost=total_cost,
        nodes_explored=nodes_explored,
        exploration_order=exploration_order,
        g_scores=g_score,
        heuristic_vals=h_cache,
    )
