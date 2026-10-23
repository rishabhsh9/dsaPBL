"""
benchmark.py - Scalability and Performance Benchmark
Benchmarks Dijkstra vs. A* search algorithms on synthetic random geometric graphs
with 100, 500, 1000, and 2000 nodes.
Measures execution time (ms) and nodes explored.
"""

import math
import random
import time
from typing import List, Dict, Tuple
from graph import DeliveryGraph
from dijkstra import dijkstra
from astar import astar


def generate_random_network(num_nodes: int,
                            avg_degree: float = 3.8,
                            map_size: float = 1000.0,
                            seed: int = 42) -> DeliveryGraph:
    """
    Generates a connected random geometric graph representing a large-scale
    city road infrastructure.

    Properties:
    - 2D spatial coordinates in [0, map_size] x [0, map_size]
    - Guaranteed full connectivity (via random spanning tree backbone)
    - Additional nearest-neighbor edges to achieve target degree
    - Edge weights >= Euclidean distances to preserve A* heuristic admissibility
    """
    rng = random.Random(seed)
    graph = DeliveryGraph()

    # 1. Place nodes randomly in 2D plane
    coords: Dict[str, Tuple[float, float]] = {}
    for i in range(num_nodes):
        node_id = f"N{i}"
        name = f"Location {i}" if i > 0 else "Central Warehouse"
        x = round(rng.uniform(0.0, map_size), 2)
        y = round(rng.uniform(0.0, map_size), 2)
        graph.add_node(node_id, name, x, y, is_warehouse=(i == 0))
        coords[node_id] = (x, y)

    node_ids = graph.get_nodes()

    # 2. Build a guaranteed connected backbone (Randomized Prim's-style MST)
    connected = {node_ids[0]}
    unconnected = set(node_ids[1:])
    added_edges = set()

    # To be fast for large graphs, sample candidates
    while unconnected:
        u = rng.choice(list(connected))
        # Find closest node in unconnected (sample up to 25 candidates for speed)
        candidates = rng.sample(list(unconnected), min(25, len(unconnected)))
        v = min(candidates, key=lambda cand: (coords[u][0] - coords[cand][0])**2 + (coords[u][1] - coords[cand][1])**2)

        eucl = graph.euclidean_distance(u, v)
        weight = round(eucl * rng.uniform(1.05, 1.25), 2)
        graph.add_edge(u, v, weight)
        added_edges.add((min(u, v), max(u, v)))

        connected.add(v)
        unconnected.remove(v)

    # 3. Add additional short edges to reach target average degree
    total_edges_target = int(num_nodes * avg_degree / 2.0)
    current_edges = graph.num_edges()

    # Grid spatial hashing for fast nearest-neighbor edge additions
    cell_size = map_size / max(1, int(math.sqrt(num_nodes) / 2))
    grid: Dict[Tuple[int, int], List[str]] = {}
    for node_id, (x, y) in coords.items():
        cell = (int(x // cell_size), int(y // cell_size))
        grid.setdefault(cell, []).append(node_id)

    attempts = 0
    max_attempts = (total_edges_target - current_edges) * 10

    while current_edges < total_edges_target and attempts < max_attempts:
        attempts += 1
        u = rng.choice(node_ids)
        ux, uy = coords[u]
        cx, cy = int(ux // cell_size), int(uy // cell_size)

        # Look in neighboring cells
        neighbor_nodes = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_nodes.extend(grid.get((cx + dx, cy + dy), []))

        if not neighbor_nodes:
            continue

        v = rng.choice(neighbor_nodes)
        if u == v:
            continue

        edge_key = (min(u, v), max(u, v))
        if edge_key not in added_edges:
            eucl = graph.euclidean_distance(u, v)
            if eucl < (map_size * 0.25):  # Only connect nearby locations
                weight = round(eucl * rng.uniform(1.05, 1.25), 2)
                graph.add_edge(u, v, weight)
                added_edges.add(edge_key)
                current_edges += 1

    return graph


def run_scalability_benchmark(sizes: List[int] = None, num_trials: int = 5, seed: int = 42) -> List[dict]:
    """
    Runs scalability benchmark comparing Dijkstra vs A* on random networks
    of 100, 500, 1000, and 2000 nodes.

    Parameters:
    - sizes: List of graph sizes (default: [100, 500, 1000, 2000])
    - num_trials: Number of random source-target pairs tested per graph size
    - seed: Random seed for reproducible experiments

    Returns:
    - List of result dictionaries containing metrics for each size
    """
    if sizes is None:
        sizes = [100, 500, 1000, 2000]

    rng = random.Random(seed)
    results = []

    print("\n" + "=" * 80)
    print("  RUNNING SCALABILITY BENCHMARK (DIJKSTRA vs A*)")
    print("=" * 80)

    for n in sizes:
        print(f"Generating synthetic graph with {n} nodes...", end="", flush=True)
        graph = generate_random_network(num_nodes=n, avg_degree=3.8, seed=seed + n)
        print(f" done ({graph.num_edges()} edges).")

        node_ids = graph.get_nodes()
        dijk_times = []
        dijk_nodes = []
        astar_times = []
        astar_nodes = []
        costs_matched = True

        for trial in range(num_trials):
            # Select two distinct nodes separated by meaningful distance
            src = rng.choice(node_ids)
            dest = rng.choice(node_ids)
            while dest == src:
                dest = rng.choice(node_ids)

            # Benchmark Dijkstra
            t0 = time.perf_counter()
            dijk_res = dijkstra(graph, start=src, target=dest)
            t1 = time.perf_counter()
            dijk_time_ms = (t1 - t0) * 1000.0

            # Benchmark A*
            t2 = time.perf_counter()
            astar_res = astar(graph, start=src, target=dest)
            t3 = time.perf_counter()
            astar_time_ms = (t3 - t2) * 1000.0

            # Verify identical optimal cost
            if abs(dijk_res.cost - astar_res.cost) > 1e-4:
                costs_matched = False

            dijk_times.append(dijk_time_ms)
            dijk_nodes.append(dijk_res.nodes_explored)
            astar_times.append(astar_time_ms)
            astar_nodes.append(astar_res.nodes_explored)

        avg_dijk_time = sum(dijk_times) / len(dijk_times)
        avg_dijk_nodes = sum(dijk_nodes) / len(dijk_nodes)
        avg_astar_time = sum(astar_times) / len(astar_times)
        avg_astar_nodes = sum(astar_nodes) / len(astar_nodes)

        explored_reduction_pct = ((avg_dijk_nodes - avg_astar_nodes) / avg_dijk_nodes) * 100.0 if avg_dijk_nodes > 0 else 0.0
        speedup = (avg_dijk_time / avg_astar_time) if avg_astar_time > 0 else 1.0

        record = {
            "num_nodes": n,
            "num_edges": graph.num_edges(),
            "dijk_time_ms": avg_dijk_time,
            "dijk_nodes": avg_dijk_nodes,
            "astar_time_ms": avg_astar_time,
            "astar_nodes": avg_astar_nodes,
            "explored_reduction_pct": explored_reduction_pct,
            "speedup": speedup,
            "costs_identical": costs_matched,
        }
        results.append(record)

        print(f"  --> Size {n:4d}: Dijkstra {avg_dijk_time:6.2f}ms ({avg_dijk_nodes:6.1f} nodes) | "
              f"A* {avg_astar_time:6.2f}ms ({avg_astar_nodes:6.1f} nodes) | "
              f"Pruned: {explored_reduction_pct:5.1f}% | "
              f"Cost Match: {costs_matched}")

    return results


def format_benchmark_table(results: List[dict]) -> str:
    """Formats benchmark results as a clean markdown/ASCII table."""
    lines = []
    lines.append("| Nodes (V) | Edges (E) | Dijkstra Time (ms) | A* Time (ms) | Speedup | Dijkstra Explored | A* Explored | Search Space Pruned | Optimal Match |")
    lines.append("|:---------:|:---------:|:------------------:|:------------:|:-------:|:-----------------:|:-----------:|:-------------------:|:-------------:|")
    for r in results:
        lines.append(
            f"| {r['num_nodes']:9d} | {r['num_edges']:9d} | {r['dijk_time_ms']:18.3f} | {r['astar_time_ms']:12.3f} | "
            f"{r['speedup']:6.2f}x | {r['dijk_nodes']:17.1f} | {r['astar_nodes']:11.1f} | "
            f"{r['explored_reduction_pct']:18.1f}% | {'YES' if r['costs_identical'] else 'NO'} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    res = run_scalability_benchmark([100, 500, 1000, 2000])
    print("\n" + format_benchmark_table(res))
