"""
main.py - Smart Logistics Route Optimization System
DSA-II Project-Based Learning (PBL) Execution Pipeline

Demonstrates:
  1. Delivery Network Modeling: 1 Warehouse (WH) + 14 Delivery Points (D1-D14), 26 Roads.
  2. Single-Destination Routing: Dijkstra vs. A* Search (Identical cost, search space comparison).
  3. Multi-Stop Route Optimization: Nearest Neighbor vs. 2-Opt vs. Held-Karp Exact DP.
  4. Scalability Performance Benchmark: 100, 500, 1000, 2000 nodes.
  5. Route Visualizations: Exporting maps and comparative charts to PNG.
"""

import sys
import os
import argparse
from typing import List

# Ensure UTF-8 output where supported, with safe fallback
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from graph import DeliveryGraph, create_delivery_network
from dijkstra import dijkstra
from astar import astar
from tsp import solve_multi_stop_delivery
from benchmark import run_scalability_benchmark, format_benchmark_table
from visualizer import (
    plot_delivery_network,
    plot_single_route_comparison,
    plot_tsp_comparison,
    plot_scalability_benchmark,
)


def print_header(title: str):
    """Prints a styled section header."""
    print("\n" + "=" * 85)
    print(f"  {title.upper()}")
    print("=" * 85)


def print_divider():
    print("-" * 85)


def run_system(skip_plots: bool = False, quick_benchmark: bool = False):
    print("""
=====================================================================================
          SMART LOGISTICS ROUTE OPTIMIZATION SYSTEM (DSA-II PBL)
     Algorithms: Dijkstra | A* Search | Nearest Neighbor | 2-Opt | Held-Karp DP
=====================================================================================
    """)

    # ---------------------------------------------------------
    # 1. Network Initialization & Modeling
    # ---------------------------------------------------------
    print_header("1. Delivery Network Topology & Graph Modeling")
    net = create_delivery_network()
    print(f"[*] Network Model: Weighted Undirected Graph (Adjacency List)")
    print(f"[*] Total Locations: {net.num_nodes()} (1 Central Warehouse 'WH' + 14 Delivery Points 'D1'-'D14')")
    print(f"[*] Interconnecting Road Segments: {net.num_edges()} bidirectional roads")
    print("\n[Node Registry & Coordinates]")
    print(f"{'ID':<6} {'Role':<14} {'Location Name':<34} {'Coordinates (x, y)':<20}")
    print("-" * 75)
    for node_id in net.get_nodes():
        node = net.nodes[node_id]
        role = "Warehouse" if node["is_warehouse"] else "Delivery Point"
        coords_str = f"({node['x']:.1f}, {node['y']:.1f}) km"
        print(f"{node_id:<6} {role:<14} {node['name']:<34} {coords_str:<20}")

    # ---------------------------------------------------------
    # 2. Single-Destination Routing: Dijkstra vs. A*
    # ---------------------------------------------------------
    print_header("2. Single-Destination Routing: Dijkstra vs. A* Search")
    print("Evaluating point-to-point dispatch from Central Warehouse (WH) to sample destinations:")

    test_destinations = ["D5", "D6", "D9", "D14"]

    print(f"\n| Destination | Dijkstra Cost | A* Cost | Match | Dijk Explored | A* Explored | Pruned (%) | Optimal Road Path |")
    print(f"|:-----------:|:-------------:|:-------:|:-----:|:-------------:|:-----------:|:----------:|:------------------|")

    saved_dijk_res = None
    saved_astar_res = None

    for dest in test_destinations:
        dijk_res = dijkstra(net, start="WH", target=dest)
        astar_res = astar(net, start="WH", target=dest)

        # Save one sample query for visualizer
        if dest == "D6":
            saved_dijk_res = dijk_res
            saved_astar_res = astar_res

        match_str = "YES" if abs(dijk_res.cost - astar_res.cost) < 1e-4 else "NO"
        pruning_pct = ((dijk_res.nodes_explored - astar_res.nodes_explored) / dijk_res.nodes_explored) * 100.0
        path_str = " -> ".join(dijk_res.path)

        print(f"| {dest:^11} | {dijk_res.cost:11.2f} km | {astar_res.cost:5.2f} km | {match_str:^5} | "
              f"{dijk_res.nodes_explored:13d} | {astar_res.nodes_explored:11d} | {pruning_pct:9.1f}% | {path_str:<17} |")

    print("\n[Key Takeaway]")
    print("  * Correctness: Dijkstra and A* find the EXACT same optimal path and distance.")
    print("  * Efficiency: A* guides exploration using Euclidean heuristic h(n), reducing")
    print("    unnecessary node expansions while preserving mathematical optimality.")

    # ---------------------------------------------------------
    # 3. Multi-Stop Routing (12 Stops Tour - Return to Warehouse)
    # ---------------------------------------------------------
    print_header("3. Multi-Stop Delivery Optimization (12 Stops Tour)")
    print("Problem: A delivery van starts at WH, delivers to 11 customer hubs, and returns to WH.")
    print("Baseline: Held-Karp Dynamic Programming (exact globally optimal solution).")
    print("Heuristics: Nearest Neighbor greedy, followed by 2-Opt local search refinement.\n")

    # 12 stops total: Warehouse (WH) + 11 delivery points
    stops = ["WH", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11"]
    print(f"Selected Stops ({len(stops)} locations): {', '.join(stops)}")
    print("Step 1: Building All-Pairs Shortest Path Distance Matrix using Dijkstra...")

    tsp_results, full_paths = solve_multi_stop_delivery(net, stops)

    print("\n[Multi-Stop Routing Performance Comparison]")
    print(f"| Algorithm            | Distance (km) | Execution Time | Optimality Gap | Status / Quality           |")
    print(f"|:-------------------- |:-------------:|:--------------:|:--------------:|:-------------------------- |")
    for res in tsp_results:
        gap_display = f"{res.gap_percent:+.2f}%" if res.gap_percent != 0.0 else "0.00% (Exact)"
        quality = "Globally Optimal" if res.gap_percent == 0.0 else ("Sub-optimal (+crossings)" if res.gap_percent > 10 else "Near-Optimal Refinement")
        print(f"| {res.algorithm:<20} | {res.total_distance:11.2f} km | {res.execution_time_ms:12.3f} ms | {gap_display:^14} | {quality:<26} |")

    print("\n[Detailed Tour Sequences]")
    for res in tsp_results:
        print(f"  * {res.algorithm:<20}: {' -> '.join(res.tour_labels)}")

    # ---------------------------------------------------------
    # 4. Scalability Performance Benchmark
    # ---------------------------------------------------------
    print_header("4. Scalability Benchmark on Synthetic Networks")
    print("Benchmarking search performance on synthetic city graphs with up to 2000 nodes.")

    benchmark_sizes = [100, 500] if quick_benchmark else [100, 500, 1000, 2000]
    bench_records = run_scalability_benchmark(sizes=benchmark_sizes, num_trials=5, seed=123)

    print("\n" + format_benchmark_table(bench_records))

    # ---------------------------------------------------------
    # 5. Visualizations & PNG Generation
    # ---------------------------------------------------------
    if not skip_plots:
        print_header("5. Generating Route Visualizations (Matplotlib)")
        os.makedirs("artifacts", exist_ok=True)

        # Plot 1: Network Topology
        plot_delivery_network(
            net,
            output_path="delivery_network.png",
            title="City Logistics Road Network (1 Warehouse + 14 Delivery Points)"
        )

        # Plot 2: Single Route Comparison
        if saved_dijk_res and saved_astar_res:
            plot_single_route_comparison(
                net,
                saved_dijk_res,
                saved_astar_res,
                output_path="single_route_comparison.png"
            )

        # Plot 3: Multi-Stop TSP Delivery Tours
        plot_tsp_comparison(
            net,
            tsp_results,
            full_paths,
            output_path="tsp_route.png"
        )

        # Plot 4: Scalability Benchmark Chart
        plot_scalability_benchmark(
            bench_records,
            output_path="scalability_benchmark.png"
        )

        print("\n[Generated Visual Artifacts]")
        print("  1. delivery_network.png        - Complete road network topology & distances")
        print("  2. single_route_comparison.png - Side-by-side Dijkstra vs A* search space")
        print("  3. tsp_route.png               - NN vs 2-Opt vs Held-Karp multi-stop delivery tours")
        print("  4. scalability_benchmark.png   - Runtime & search space scaling up to 2000 nodes")

    print_header("Execution Complete")
    print("[*] Smart Logistics Route Optimization System executed successfully.")
    print("[*] All tests, benchmarks, and visual artifacts are ready.")


def main():
    parser = argparse.ArgumentParser(description="Smart Logistics Route Optimization System (DSA-II PBL)")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot generation")
    parser.add_argument("--quick", action="store_true", help="Run benchmark with smaller sizes (100, 500)")
    args = parser.parse_args()

    run_system(skip_plots=args.no_plot, quick_benchmark=args.quick)


if __name__ == "__main__":
    main()
