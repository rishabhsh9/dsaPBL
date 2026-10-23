# Smart Logistics Route Optimization System
**DSA-II Project-Based Learning (PBL) Project**

> ⚠️ **Note:** Parts of this project's code and documentation were developed with the assistance of AI tools for educational and academic PBL demonstration purposes.

---

## 📌 About The Project

This is a student project for **DSA-II (Data Structures and Algorithms II)**. It simulates a smart delivery logistics system for a city.

The goal is to find the best routes for delivery vans:
1. **Single-destination:** Finding the shortest path from a warehouse to a delivery point using **Dijkstra's** and **A\*** algorithms.
2. **Multi-stop delivery (TSP):** Visiting 12 delivery stops and returning to the warehouse using **Nearest Neighbor**, **2-Opt**, and the exact **Held-Karp** algorithm.
3. **Performance tests:** Testing search speed and nodes explored on random networks with 100, 500, 1000, and 2000 nodes.
4. **Visual plots:** Saving route maps and comparison graphs using `matplotlib`.

---

## 🗺️ The Delivery Network

- **1 Warehouse (`WH`)** + **14 Delivery Points (`D1` to `D14`)** = 15 total locations.
- **26 bidirectional roads** with distances given in kilometers.
- Represented using an **Adjacency List** for fast lookup and low memory usage.

---

## ⚙️ Algorithms Used

| Algorithm | Used For | Time Complexity | Notes |
|---|---|:---:|---|
| **Dijkstra** | Shortest path | $O((V + E) \log V)$ | Explores uniformly in all directions. |
| **A\* Search** | Shortest path | $O((V + E) \log V)$ | Uses Euclidean distance to reach the goal faster with fewer node visits. |
| **Nearest Neighbor** | Multi-stop tour | $O(N^2)$ | Greedy approach: always visits the closest unvisited stop. |
| **2-Opt** | Tour improvement | $O(k \cdot N^2)$ | Removes crossing paths from the tour. |
| **Held-Karp (DP)** | Exact TSP baseline | $O(N^2 \cdot 2^N)$ | Uses bitmask dynamic programming to find the true minimum tour. |

---

## 🚀 How to Run

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the program:**
   ```bash
   python main.py
   ```

3. **Optional flags:**
   - Quick test: `python main.py --quick`
   - Without plots: `python main.py --no-plot`

---

## 📊 Sample Output

### 1. Dijkstra vs. A* (Same cost, fewer nodes explored)
| Destination | Dijkstra Cost | A* Cost | Cost Match? | Dijkstra Explored | A* Explored | Nodes Saved |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D5** | 42.0 km | 42.0 km | YES | 13 nodes | 3 nodes | **76.9%** |
| **D6** | 44.0 km | 44.0 km | YES | 14 nodes | 3 nodes | **78.6%** |
| **D9** | 34.0 km | 34.0 km | YES | 8 nodes | 3 nodes | **62.5%** |
| **D14** | 42.0 km | 42.0 km | YES | 12 nodes | 4 nodes | **66.7%** |

### 2. Multi-Stop Route (12 stops, return to Warehouse)
| Algorithm | Distance | Time | Optimality Gap |
|---|:---:|:---:|:---:|
| **Nearest Neighbor** | 276.0 km | 0.02 ms | +0.91% |
| **2-Opt Improvement** | 276.0 km | 0.02 ms | +0.91% |
| **Held-Karp (Exact DP)** | **273.5 km** | 40.5 ms | **0.00% (Best)** |

---

## 📁 Files in This Project

- `graph.py` - Graph structure and city network data.
- `dijkstra.py` - Dijkstra's algorithm using min-heap.
- `astar.py` - A* algorithm with Euclidean distance.
- `tsp.py` - TSP algorithms (Nearest Neighbor, 2-Opt, Held-Karp).
- `benchmark.py` - Tests on 100, 500, 1000, 2000 node graphs.
- `visualizer.py` - Makes and saves charts as PNG images.
- `main.py` - Runs everything and displays results.
- `requirements.txt` - Dependencies (`matplotlib`).
