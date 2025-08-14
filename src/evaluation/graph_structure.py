# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeAlias


@dataclass
class Graph:
    """
    @brief Represents an undirected input graph with vertex or edge weights
    """

    # The number of vertices in the graph.
    n: int

    # The number of edges in the graph.
    m: int

    # A list of pairs of integers -- each representing an edge of the graph.
    edges: list[tuple[int, int]]

    # The weight of each edge, where `edge_weights[idx]` is the weight of the
    # `idx`-th edge, in the `edges` list.
    edge_weights: list[int] | None = None

    # Map of vertex -> weight.
    vertex_weights: defaultdict[int, int] | None = None

    # Map of vertex -> { set of indices of edges in which it appears }
    _incidence: dict[int, set[int]] = None

    # Map of vertex -> { set of vertices to which it is adjacent }
    _adj: dict[int, set[int]] = None

    def __post_init__(self):

        # Ensure the edge weights are fine.
        assert len(self.edges) == self.m
        if self.edge_weights is not None:
            assert all(self.edge_weights[idx] is not None for idx in range(len(self.edges)))

        # Populate the adjacency and incidence maps.
        adj = defaultdict(set)
        incidence = defaultdict(set)
        for idx, (u, v) in enumerate(self.edges):
            adj[u].add(v)
            adj[v].add(u)
            incidence[u].add(idx)
            incidence[v].add(idx)
        self._adj = adj
        self._incidence = incidence

    def neighbors(self, v: int) -> set[int]:
        return self._adj[v]

    def induced_subgraph(self, vertices: list[int]) -> "Graph":
        """
        @brief Creates the induced subgraph over the given set of vertices, keeping the original
               vertex labels.
        @param vertices A list of vertex indices (original labels) to include in the subgraph.
        @return A new Graph object representing the induced subgraph. This graph is a 'view'
                in the sense that it uses the original vertex labels, but it is a new Graph
                object and modifications to it will not affect the original graph.
        """

        induced_vertices_set = set(vertices)
        subgraph_n = len(induced_vertices_set)

        # Collect the vertex weights, if relevant.
        subgraph_vertex_weights = None
        if self.vertex_weights is not None:
            subgraph_vertex_weights = defaultdict(int)
            for v_label in induced_vertices_set:
                w = self.vertex_weights.get(v_label, None)
                if w is not None:
                    subgraph_vertex_weights[v_label] = w

        # Collect the edge weights, if relevant.
        subgraph_edges = []
        subgraph_edge_weights = [] if self.edge_weights is not None else None
        processed_edge_indices = set()
        subgraph_m = 0

        for v in induced_vertices_set:

            for edge_idx in self._incidence.get(v, set()):
                if edge_idx in processed_edge_indices:
                    continue  # This edge has already been processed from its other endpoint

                u, w = self.edges[edge_idx]

                # An edge (u, w) is part of the induced subgraph if both u and w are in
                # the set of induced vertices.
                if u in induced_vertices_set and w in induced_vertices_set:
                    subgraph_edges.append((u, w))
                    if self.edge_weights is not None:
                        subgraph_edge_weights.append(self.edge_weights[edge_idx])
                    subgraph_m += 1

                # Mark the edge as processed so it's not added again (even if not induced)
                processed_edge_indices.add(edge_idx)

        return Graph(
            n=subgraph_n,
            m=subgraph_m,
            edges=subgraph_edges,
            edge_weights=subgraph_edge_weights,
            vertex_weights=subgraph_vertex_weights,
        )


# Tree decomposition bag. Represents a set of vertices from the original graph.
Bag: TypeAlias = set[int]
