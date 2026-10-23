"""
tsp.py - Multi-Stop Delivery Route Optimization (TSP)
Solves the Traveling Salesperson Problem (TSP) for delivery vehicles:
  1. Builds distance matrix using Dijkstra's algorithm on the road network.
  2. Nearest Neighbor (greedy heuristic) - O(N^2)
  3. 2-Opt Local Search Improvement - O(k * N^2)
  4. Held-Karp Algorithm (Exact bitmask DP) - O(N^2 * 2^N) baseline

Starts at Warehouse (WH), visits all required delivery points, and returns to WH.
"""

import time
from typing import List, Dict, Tuple, Optional
from graph import DeliveryGraph
from dijkstra import dijkstra


class TSPResult:
    """Stores the evaluation metrics of a TSP routing algorithm."""

    def __init__(self,
                 algorithm: str,
                 tour_indices: List[int],
                 tour_labels: List[str],
                 total_distance: float,
                 execution_time_ms: float,
                 gap_percent: float = 0.0):
        self.algorithm = algorithm
        self.tour_indices = tour_indices
        self.tour_labels = tour_labels
        self.total_distance = total_distance
        self.execution_time_ms = execution_time_ms
        self.gap_percent = gap_percent

    def __repr__(self) -> str:
        route_str = " -> ".join(self.tour_labels)
        return (f"[{self.algorithm}] Dist: {self.total_distance:.2f} km | "
                f"Time: {self.execution_time_ms:.2f} ms | "
                f"Gap: {self.gap_percent:+.2f}% | "
                f"Tour: {route_str}")


def build_distance_matrix(graph: DeliveryGraph, stops: List[str]) -> Tuple[List[List[float]], Dict[Tuple[int, int], List[str]]]:
    """
    Computes all-pairs shortest paths between the requested delivery stops
    using Dijkstra's algorithm.

    Parameters:
    - graph: DeliveryGraph instance
    - stops: List of node IDs (e.g., ['WH', 'D1', 'D2', ...])

    Returns:
    - dist_matrix: N x N 2D list where dist_matrix[i][j] is the road distance
    - full_paths: Dict mapping (i, j) to the sequence of graph nodes traversed
    """
    n = len(stops)
    dist_matrix = [[0.0] * n for _ in range(n)]
    full_paths: Dict[Tuple[int, int], List[str]] = {}

    for i, origin in enumerate(stops):
        dijk_res = dijkstra(graph, start=origin, target=None)

        for j, dest in enumerate(stops):
            if i == j:
                dist_matrix[i][j] = 0.0
                full_paths[(i, j)] = [origin]
            else:
                dist_matrix[i][j] = dijk_res.distances[dest]

                # Reconstruct path from predecessors
                path = []
                curr = dest
                while curr is not None:
                    path.append(curr)
                    curr = dijk_res.parents[curr]
                path.reverse()
                full_paths[(i, j)] = path

    return dist_matrix, full_paths


def calculate_tour_distance(tour: List[int], dist_matrix: List[List[float]]) -> float:
    """Calculates total closed tour distance."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += dist_matrix[tour[i]][tour[i + 1]]
    return total


def nearest_neighbor_tsp(stops: List[str], dist_matrix: List[List[float]]) -> TSPResult:
    """
    Nearest Neighbor greedy heuristic.
    Starts at Warehouse (index 0), repeatedly moves to the closest unvisited stop,
    and returns to Warehouse.

    Time Complexity: O(N^2)
    Space Complexity: O(N)
    """
    start_time = time.perf_counter()
    n = len(stops)
    visited = [False] * n

    current = 0
    visited[current] = True
    tour = [current]

    for _ in range(n - 1):
        next_node = -1
        min_dist = float("inf")
        for candidate in range(n):
            if not visited[candidate] and dist_matrix[current][candidate] < min_dist:
                min_dist = dist_matrix[current][candidate]
                next_node = candidate

        visited[next_node] = True
        tour.append(next_node)
        current = next_node

    # Complete the loop back to start (Warehouse)
    tour.append(0)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    total_dist = calculate_tour_distance(tour, dist_matrix)
    tour_labels = [stops[idx] for idx in tour]

    return TSPResult(
        algorithm="Nearest Neighbor",
        tour_indices=tour,
        tour_labels=tour_labels,
        total_distance=total_dist,
        execution_time_ms=elapsed_ms,
    )


def two_opt_improvement(stops: List[str], dist_matrix: List[List[float]], initial_tour: List[int]) -> TSPResult:
    """
    2-Opt local search improvement.
    Takes an initial tour (e.g. from Nearest Neighbor) and systematically removes
    crossing or inefficient edge pairs by reversing sub-segments: tour[i:j+1].

    Time Complexity: O(k * N^2) where k is the number of improvement passes.
    Space Complexity: O(N)
    """
    start_time = time.perf_counter()
    tour = list(initial_tour)
    n = len(stops)

    improved = True
    max_passes = 100
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        # We keep tour[0] (Warehouse) and tour[-1] (Warehouse) fixed
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Current edges: (i-1 -> i) and (j -> j+1)
                # Proposed edges: (i-1 -> j) and (i -> j+1)
                current_cost = dist_matrix[tour[i - 1]][tour[i]] + dist_matrix[tour[j]][tour[j + 1]]
                proposed_cost = dist_matrix[tour[i - 1]][tour[j]] + dist_matrix[tour[i]][tour[j + 1]]

                if proposed_cost < current_cost - 1e-9:
                    # Reverse segment from i to j
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
                    break
            if improved:
                break

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    total_dist = calculate_tour_distance(tour, dist_matrix)
    tour_labels = [stops[idx] for idx in tour]

    return TSPResult(
        algorithm="2-Opt Improvement",
        tour_indices=tour,
        tour_labels=tour_labels,
        total_distance=total_dist,
        execution_time_ms=elapsed_ms,
    )


def held_karp_exact_tsp(stops: List[str], dist_matrix: List[List[float]]) -> TSPResult:
    """
    Held-Karp dynamic programming algorithm using bitmask representation.
    Provides the provably exact, mathematically optimal TSP tour.

    State: dp(mask, u) = min cost to start at 0, visit subset of nodes in mask, ending at u.
    Bitmask: bit i represents whether stop i has been visited.

    Time Complexity: O(N^2 * 2^N)
    Space Complexity: O(N * 2^N)
    """
    start_time = time.perf_counter()
    n = len(stops)

    # memo[(mask, u)] = (min_cost, parent_node)
    # mask: integer bitmask where bit i is 1 if node i is in the subset
    memo: Dict[Tuple[int, int], Tuple[float, int]] = {}

    # Base case: starting at warehouse (node 0)
    # Mask with only bit 0 set: 1 << 0
    memo[(1 << 0, 0)] = (0.0, -1)

    # Iterate through subset sizes from 2 to n
    for size in range(2, n + 1):
        # Generate all masks of given size that include node 0 (bit 0 is 1)
        for mask in range(1, 1 << n):
            if not (mask & 1):  # Must contain node 0
                continue
            if bin(mask).count('1') != size:
                continue

            # Consider each candidate end-node u in the mask (u != 0)
            for u in range(1, n):
                if not (mask & (1 << u)):
                    continue

                prev_mask = mask ^ (1 << u)
                min_cost = float("inf")
                best_parent = -1

                # Find predecessor p in prev_mask
                for p in range(n):
                    if prev_mask & (1 << p):
                        if (prev_mask, p) in memo:
                            cost = memo[(prev_mask, p)][0] + dist_matrix[p][u]
                            if cost < min_cost:
                                min_cost = cost
                                best_parent = p

                if best_parent != -1:
                    memo[(mask, u)] = (min_cost, best_parent)

    # Complete the tour by returning to warehouse (node 0)
    full_mask = (1 << n) - 1
    optimal_cost = float("inf")
    last_node = -1

    for u in range(1, n):
        if (full_mask, u) in memo:
            total_tour = memo[(full_mask, u)][0] + dist_matrix[u][0]
            if total_tour < optimal_cost:
                optimal_cost = total_tour
                last_node = u

    # Backtrack to reconstruct the optimal tour sequence
    curr_mask = full_mask
    curr_node = last_node

    path_backwards = []
    while curr_node != -1:
        path_backwards.append(curr_node)
        prev_node = memo[(curr_mask, curr_node)][1]
        curr_mask = curr_mask ^ (1 << curr_node)
        curr_node = prev_node

    path_backwards.reverse()
    tour = list(path_backwards)
    tour.append(0)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    tour_labels = [stops[idx] for idx in tour]

    return TSPResult(
        algorithm="Held-Karp (Exact DP)",
        tour_indices=tour,
        tour_labels=tour_labels,
        total_distance=optimal_cost,
        execution_time_ms=elapsed_ms,
        gap_percent=0.0,
    )


def solve_multi_stop_delivery(graph: DeliveryGraph, stops: List[str]) -> Tuple[List[TSPResult], Dict[Tuple[int, int], List[str]]]:
    """
    Executes the full multi-stop routing pipeline:
      1. Build all-pairs distance matrix via Dijkstra
      2. Run Nearest Neighbor heuristic
      3. Run 2-Opt local search improvement
      4. Run Held-Karp exact DP baseline
      5. Calculate optimality gaps relative to Held-Karp
    """
    dist_matrix, full_paths = build_distance_matrix(graph, stops)

    # 1. Nearest Neighbor
    nn_res = nearest_neighbor_tsp(stops, dist_matrix)

    # 2. 2-Opt Improvement on NN tour
    two_opt_res = two_opt_improvement(stops, dist_matrix, nn_res.tour_indices)

    # 3. Held-Karp Exact DP Baseline
    hk_res = held_karp_exact_tsp(stops, dist_matrix)

    # Compute Optimality Gaps (%)
    exact_cost = hk_res.total_distance
    nn_res.gap_percent = ((nn_res.total_distance - exact_cost) / exact_cost) * 100.0
    two_opt_res.gap_percent = ((two_opt_res.total_distance - exact_cost) / exact_cost) * 100.0
    hk_res.gap_percent = 0.0

    return [nn_res, two_opt_res, hk_res], full_paths
