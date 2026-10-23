"""
visualizer.py - Route and Network Visualizations using Matplotlib
Generates publication-quality charts and network maps for the DSA-II PBL project:
  1. delivery_network.png: Complete city delivery network topology
  2. single_route_comparison.png: Side-by-side Dijkstra vs A* search frontiers
  3. tsp_route.png: Comparison of Nearest Neighbor, 2-Opt, and Held-Karp delivery tours
  4. scalability_benchmark.png: Performance scaling across 100 to 2000 nodes
"""

import os
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/terminal execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from graph import DeliveryGraph
from dijkstra import DijkstraResult
from astar import AStarResult
from tsp import TSPResult


def plot_delivery_network(graph: DeliveryGraph,
                          output_path: str = "delivery_network.png",
                          highlight_path: Optional[List[str]] = None,
                          title: str = "Smart Logistics City Delivery Network (1 WH + 14 Delivery Points)"):
    """
    Renders the full road network with nodes, road distances, and warehouse designation.
    """
    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
    fig.patch.set_facecolor("#0F172A")  # Modern dark slate background
    ax.set_facecolor("#1E293B")

    # Draw roads (edges)
    drawn_edges = set()
    for u in graph.get_nodes():
        x1, y1 = graph.get_coordinates(u)
        for v, weight in graph.get_neighbors(u):
            edge_key = (min(u, v), max(u, v))
            if edge_key not in drawn_edges:
                drawn_edges.add(edge_key)
                x2, y2 = graph.get_coordinates(v)

                # Check if this edge is part of a highlighted route
                is_highlighted = False
                if highlight_path:
                    for k in range(len(highlight_path) - 1):
                        if (highlight_path[k] == u and highlight_path[k+1] == v) or \
                           (highlight_path[k] == v and highlight_path[k+1] == u):
                            is_highlighted = True
                            break

                edge_color = "#38BDF8" if is_highlighted else "#475569"
                edge_width = 3.5 if is_highlighted else 1.5
                edge_alpha = 1.0 if is_highlighted else 0.6
                edge_style = "-" if is_highlighted else "--"

                ax.plot([x1, x2], [y1, y2], color=edge_color, lw=edge_width,
                        alpha=edge_alpha, linestyle=edge_style, zorder=2)

                # Distance label at midpoint
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                ax.text(mx, my, f"{weight}k", color="#94A3B8", fontsize=7.5,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#0F172A", edgecolor="#334155", lw=0.6, alpha=0.85),
                        zorder=3)

    # Draw nodes
    for node_id in graph.get_nodes():
        x, y = graph.get_coordinates(node_id)
        is_wh = graph.nodes[node_id]["is_warehouse"]
        name = graph.nodes[node_id]["name"]

        if is_wh:
            # Warehouse: Prominent gold/red star
            ax.scatter(x, y, s=550, color="#EF4444", edgecolor="#FACC15", lw=2.5, marker="*", zorder=6)
            ax.annotate(f"{node_id}\n(Warehouse)", (x, y), xytext=(0, -22), textcoords="offset points",
                        color="#FDE047", weight="bold", fontsize=10, ha="center", zorder=7)
        else:
            # Delivery Point: Teal circle
            ax.scatter(x, y, s=260, color="#0EA5E9", edgecolor="#E2E8F0", lw=1.8, marker="o", zorder=4)
            ax.annotate(f"{node_id}", (x, y), xytext=(0, 0), textcoords="offset points",
                        color="#FFFFFF", weight="bold", fontsize=8.5, ha="center", va="center", zorder=5)
            # Subtle location name
            ax.annotate(name, (x, y), xytext=(0, 14), textcoords="offset points",
                        color="#CBD5E1", fontsize=7.5, ha="center", zorder=5)

    ax.set_title(title, color="#F8FAFC", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("East-West Coordinate (km)", color="#94A3B8", fontsize=10)
    ax.set_ylabel("North-South Coordinate (km)", color="#94A3B8", fontsize=10)
    ax.tick_params(colors="#64748B")
    ax.grid(True, linestyle=":", color="#334155", alpha=0.5)

    # Custom Legend
    wh_patch = mpatches.Patch(facecolor="#EF4444", edgecolor="#FACC15", label="Warehouse (WH)")
    dp_patch = mpatches.Patch(facecolor="#0EA5E9", edgecolor="#FFFFFF", label="Delivery Points (D1-D14)")
    road_patch = mpatches.Patch(facecolor="#475569", label="Road Network (26 Roads)")
    ax.legend(handles=[wh_patch, dp_patch, road_patch], loc="upper left",
              facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [Visualization] Saved network map -> {output_path}")


def plot_single_route_comparison(graph: DeliveryGraph,
                                 dijk_res: DijkstraResult,
                                 astar_res: AStarResult,
                                 output_path: str = "single_route_comparison.png"):
    """
    Renders side-by-side comparison of Dijkstra vs A* showing:
      1. Explored nodes frontier (uninformed circular expansion vs directed cone)
      2. Final optimal path (demonstrating identical cost)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=200)
    fig.patch.set_facecolor("#0F172A")

    for ax, res, algo_name, accent_color in [
        (ax1, dijk_res, "Dijkstra's Algorithm (Min-Heap)", "#F97316"),
        (ax2, astar_res, "A* Algorithm (Euclidean Heuristic)", "#10B981")
    ]:
        ax.set_facecolor("#1E293B")
        explored_set = set(res.exploration_order)
        path_set = set(res.path)
        path_edges = set()
        for i in range(len(res.path) - 1):
            path_edges.add((min(res.path[i], res.path[i+1]), max(res.path[i], res.path[i+1])))

        # Draw background roads
        for u in graph.get_nodes():
            x1, y1 = graph.get_coordinates(u)
            for v, weight in graph.get_neighbors(u):
                if u < v:
                    x2, y2 = graph.get_coordinates(v)
                    is_path = (min(u, v), max(u, v)) in path_edges
                    color = accent_color if is_path else "#334155"
                    lw = 4.0 if is_path else 1.2
                    alpha = 1.0 if is_path else 0.4
                    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=3 if is_path else 1)

        # Draw nodes with exploration status
        for node_id in graph.get_nodes():
            x, y = graph.get_coordinates(node_id)
            is_wh = (node_id == res.start)
            is_target = (node_id == res.target)
            is_on_path = node_id in path_set
            was_explored = node_id in explored_set

            if is_wh:
                node_color = "#EF4444"
                edge_col = "#FACC15"
                size = 400
                marker = "*"
            elif is_target:
                node_color = "#8B5CF6"
                edge_col = "#FFFFFF"
                size = 350
                marker = "D"
            elif is_on_path:
                node_color = accent_color
                edge_col = "#FFFFFF"
                size = 250
                marker = "o"
            elif was_explored:
                node_color = "#EAB308"  # Yellow/amber for explored
                edge_col = "#713F12"
                size = 180
                marker = "o"
            else:
                node_color = "#475569"  # Unvisited gray
                edge_col = "#1E293B"
                size = 120
                marker = "o"

            ax.scatter(x, y, s=size, color=node_color, edgecolor=edge_col, lw=1.5, marker=marker, zorder=4)
            label_col = "#FFFFFF" if was_explored or is_wh or is_target else "#94A3B8"
            ax.annotate(node_id, (x, y), xytext=(0, 0), textcoords="offset points",
                        color=label_col, weight="bold", fontsize=8, ha="center", va="center", zorder=5)

        # Title and stats box
        path_str = " -> ".join(res.path)
        stats_text = (f"Distance: {res.cost:.2f} km\n"
                      f"Nodes Explored: {res.nodes_explored} / {graph.num_nodes()}\n"
                      f"Path: {path_str}")

        ax.set_title(f"{algo_name}\nNodes Explored: {res.nodes_explored}",
                     color="#F8FAFC", fontsize=12, weight="bold", pad=10)
        ax.text(0.03, 0.04, stats_text, transform=ax.transAxes, color="#F1F5F9", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#0F172A", edgecolor=accent_color, lw=1.2, alpha=0.9),
                zorder=10)

        ax.set_xlabel("X (km)", color="#94A3B8", fontsize=9)
        ax.set_ylabel("Y (km)", color="#94A3B8", fontsize=9)
        ax.tick_params(colors="#64748B")
        ax.grid(True, linestyle=":", color="#334155", alpha=0.5)

    plt.suptitle(f"Single-Destination Routing Comparison: {dijk_res.start} -> {dijk_res.target}\n"
                 f"Both Algorithms Guarantee Exact Optimal Path ({dijk_res.cost:.2f} km)",
                 color="#F8FAFC", fontsize=14, weight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [Visualization] Saved single-route comparison -> {output_path}")


def plot_tsp_comparison(graph: DeliveryGraph,
                        tsp_results: List[TSPResult],
                        full_paths: Dict[Tuple[int, int], List[str]],
                        output_path: str = "tsp_route.png"):
    """
    Renders 3-panel comparison of Multi-Stop Delivery Tours:
      1. Nearest Neighbor Heuristic
      2. 2-Opt Local Search Improvement
      3. Held-Karp Exact DP Baseline
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), dpi=200)
    fig.patch.set_facecolor("#0F172A")

    palette = ["#F59E0B", "#06B6D4", "#10B981"]

    for ax, tsp_res, color in zip(axes, tsp_results, palette):
        ax.set_facecolor("#1E293B")

        # Draw road network in background
        for u in graph.get_nodes():
            x1, y1 = graph.get_coordinates(u)
            for v, _ in graph.get_neighbors(u):
                if u < v:
                    x2, y2 = graph.get_coordinates(v)
                    ax.plot([x1, x2], [y1, y2], color="#334155", lw=1.0, alpha=0.35, zorder=1)

        # Trace the concrete vehicle trajectory along actual roads
        tour = tsp_res.tour_indices
        for idx in range(len(tour) - 1):
            i, j = tour[idx], tour[idx + 1]
            road_segment = full_paths.get((i, j), [tsp_res.tour_labels[idx], tsp_res.tour_labels[idx+1]])

            # Draw trajectory along road vertices
            for k in range(len(road_segment) - 1):
                u, v = road_segment[k], road_segment[k + 1]
                x1, y1 = graph.get_coordinates(u)
                x2, y2 = graph.get_coordinates(v)
                ax.plot([x1, x2], [y1, y2], color=color, lw=2.8, alpha=0.9, zorder=3)

                # Add direction arrow in middle of segment
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dx, dy = (x2 - x1) * 0.1, (y2 - y1) * 0.1
                ax.arrow(mx - dx/2, my - dy/2, dx, dy, shape="full", lw=0, length_includes_head=True,
                         head_width=1.2, color=color, zorder=4)

        # Draw stop nodes
        for order_idx, node_label in enumerate(tsp_res.tour_labels[:-1]):
            x, y = graph.get_coordinates(node_label)
            is_wh = (node_label == "WH")

            if is_wh:
                ax.scatter(x, y, s=450, color="#EF4444", edgecolor="#FACC15", lw=2, marker="*", zorder=6)
                ax.annotate("WH (Start/End)", (x, y), xytext=(0, -18), textcoords="offset points",
                            color="#FDE047", weight="bold", fontsize=9, ha="center", zorder=7)
            else:
                ax.scatter(x, y, s=240, color="#0EA5E9", edgecolor="#FFFFFF", lw=1.5, marker="o", zorder=5)
                # Show stop order sequence number
                ax.annotate(f"{node_label}\n[#{order_idx}]", (x, y), xytext=(0, 14), textcoords="offset points",
                            color="#FFFFFF", weight="bold", fontsize=7.5, ha="center", zorder=7)

        # Stats Card
        gap_text = f"Gap: {tsp_res.gap_percent:+.2f}%" if tsp_res.gap_percent != 0.0 else "Optimal Baseline (0.0%)"
        info = (f"Algorithm: {tsp_res.algorithm}\n"
                f"Total Distance: {tsp_res.total_distance:.2f} km\n"
                f"Execution Time: {tsp_res.execution_time_ms:.2f} ms\n"
                f"Optimality: {gap_text}")

        ax.set_title(f"{tsp_res.algorithm}\nDistance: {tsp_res.total_distance:.2f} km",
                     color="#F8FAFC", fontsize=12, weight="bold", pad=10)
        ax.text(0.04, 0.05, info, transform=ax.transAxes, color="#F8FAFC", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#0F172A", edgecolor=color, lw=1.2, alpha=0.9),
                zorder=10)

        ax.tick_params(colors="#64748B")
        ax.grid(True, linestyle=":", color="#334155", alpha=0.4)

    plt.suptitle("Multi-Stop Delivery Route Optimization (12 Stops Tour - Return to Warehouse)\n"
                 "Comparison of Heuristic vs 2-Opt Improvement vs Exact Held-Karp Baseline",
                 color="#F8FAFC", fontsize=14, weight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [Visualization] Saved TSP tour comparison -> {output_path}")


def plot_scalability_benchmark(results: List[dict], output_path: str = "scalability_benchmark.png"):
    """
    Renders 2-panel performance scaling plots:
      1. Execution Time (ms) vs Graph Size (V)
      2. Nodes Explored vs Graph Size (V) showing A* search space pruning
    """
    nodes = [r["num_nodes"] for r in results]
    dijk_time = [r["dijk_time_ms"] for r in results]
    astar_time = [r["astar_time_ms"] for r in results]
    dijk_explored = [r["dijk_nodes"] for r in results]
    astar_explored = [r["astar_nodes"] for r in results]
    pruned = [r["explored_reduction_pct"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=200)
    fig.patch.set_facecolor("#0F172A")

    # --- Plot 1: Execution Time (ms) ---
    ax1.set_facecolor("#1E293B")
    ax1.plot(nodes, dijk_time, marker="o", lw=2.5, color="#F97316", label="Dijkstra's (Min-Heap)")
    ax1.plot(nodes, astar_time, marker="s", lw=2.5, color="#10B981", label="A* (Euclidean Heuristic)")
    ax1.set_title("Execution Time Scaling (ms)", color="#F8FAFC", fontsize=13, weight="bold", pad=12)
    ax1.set_xlabel("Number of Nodes |V|", color="#94A3B8", fontsize=10)
    ax1.set_ylabel("Execution Time (ms)", color="#94A3B8", fontsize=10)
    ax1.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=9.5)
    ax1.tick_params(colors="#64748B")
    ax1.grid(True, linestyle=":", color="#334155", alpha=0.6)

    # --- Plot 2: Nodes Explored (Search Space) ---
    ax2.set_facecolor("#1E293B")
    ax2.plot(nodes, dijk_explored, marker="o", lw=2.5, color="#F97316", label="Dijkstra Explored")
    ax2.plot(nodes, astar_explored, marker="s", lw=2.5, color="#10B981", label="A* Explored")
    ax2.fill_between(nodes, astar_explored, dijk_explored, color="#10B981", alpha=0.15, label="Search Space Pruned")

    for x_val, y_val, p_val in zip(nodes, astar_explored, pruned):
        ax2.annotate(f"-{p_val:.1f}%", (x_val, y_val), xytext=(0, -18), textcoords="offset points",
                     color="#34D399", weight="bold", fontsize=9, ha="center")

    ax2.set_title("Nodes Explored (Search Space Efficiency)", color="#F8FAFC", fontsize=13, weight="bold", pad=12)
    ax2.set_xlabel("Number of Nodes |V|", color="#94A3B8", fontsize=10)
    ax2.set_ylabel("Nodes Popped from Priority Queue", color="#94A3B8", fontsize=10)
    ax2.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=9.5)
    ax2.tick_params(colors="#64748B")
    ax2.grid(True, linestyle=":", color="#334155", alpha=0.6)

    plt.suptitle("Algorithm Scalability Benchmark: Dijkstra vs. A* (100 - 2000 Nodes)",
                 color="#F8FAFC", fontsize=14, weight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [Visualization] Saved scalability benchmark chart -> {output_path}")
