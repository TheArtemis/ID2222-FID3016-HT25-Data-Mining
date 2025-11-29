import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from miner.core.spectra.cluster_machine import ClusterMachine
from miner.core.spectra.eigen_plotter import EigenPlotter
from miner.colors import get_palette_dict


class ClusterPlotter:
    """Class for plotting clustering results and eigenvalue analysis."""

    def __init__(self, cluster_machine: ClusterMachine):
        """
        Initialize the plotter with a ClusterMachine instance.

        Args:
            cluster_machine: The ClusterMachine instance containing the data to plot
        """
        self.cluster_machine = cluster_machine

        if (
            self.cluster_machine.eigenvalues is None
            or self.cluster_machine.eigenvectors is None
        ):
            self.cluster_machine.compute_eigenvalues()

        self.eigen_plotter = EigenPlotter(
            self.cluster_machine.eigenvalues, self.cluster_machine.eigenvectors
        )

        self.logger = logging.getLogger(__name__)

    def plot_graph_structure(
        self,
        output_dir: Path,
        filename: str = "graph_structure.png",
        layout: str = "spring",
        node_size: int = 50,
        figsize: tuple[int, int] = (12, 10),
        max_nodes: int | None = None,
        custom_colors: dict[int, str] | list[str] | None = None,
        cmap: str | None = None,
    ):
        """
        Plot the graph structure with nodes colored by cluster assignment.

        Args:
            output_dir: Directory where the plot will be saved
            filename: Name of the output file
            layout: Layout algorithm to use ('spring', 'kamada_kawai', 'circular', 'spectral')
            node_size: Size of nodes in the plot
            figsize: Figure size (width, height)
            max_nodes: Maximum number of nodes to plot (for large graphs). If None, plots all nodes.
            custom_colors: Custom colors for clusters. Can be:
                - dict mapping cluster ID to color (e.g., {0: 'red', 1: 'blue'})
                - list of colors (one per cluster in order)
            cmap: Colormap name to use (e.g., 'tab10', 'Set3', 'viridis').
                  If custom_colors is provided, this is ignored.
        """
        if self.cluster_machine.latest_clusters is None:
            self.logger.error(
                "Before plotting, you need to cluster the data first: run .cluster() first"
            )
            return

        # Convert sparse matrix to NetworkX graph
        graph_matrix = self.cluster_machine.graph
        G = nx.from_scipy_sparse_array(graph_matrix)

        # Validate that cluster assignments match graph nodes
        if len(self.cluster_machine.latest_clusters) != G.number_of_nodes():
            self.logger.error(
                f"Mismatch: cluster assignments ({len(self.cluster_machine.latest_clusters)}) "
                f"don't match graph nodes ({G.number_of_nodes()})"
            )
            return

        # Optionally sample nodes for large graphs
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            self.logger.warning(
                f"Graph has {G.number_of_nodes()} nodes, sampling {max_nodes} nodes for visualization"
            )
            # Sample nodes (prefer nodes with higher degree for better visualization)
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[
                :max_nodes
            ]
            sampled_nodes = [node for node, _ in top_nodes]
            G = G.subgraph(sampled_nodes).copy()
            # Map cluster colors to sampled nodes (preserve node order in subgraph)
            cluster_colors = [
                self.cluster_machine.latest_clusters[node] for node in G.nodes()
            ]
        else:
            # Map cluster colors to nodes in graph order
            # NetworkX from_scipy_sparse_array preserves node indices (0, 1, 2, ...)
            cluster_colors = [
                self.cluster_machine.latest_clusters[node] for node in G.nodes()
            ]

        # Choose layout algorithm
        layout_funcs = {
            "spring": nx.spring_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
            "spectral": nx.spectral_layout,
        }

        if layout not in layout_funcs:
            self.logger.warning(f"Unknown layout '{layout}', using 'spring' instead")
            layout = "spring"

        try:
            if layout == "kamada_kawai" and G.number_of_nodes() > 100:
                self.logger.warning(
                    "Kamada-Kawai layout is slow for large graphs, using spring layout instead"
                )
                pos = nx.spring_layout(G, k=1, iterations=50)
            else:
                pos = layout_funcs[layout](G)
        except Exception as e:
            self.logger.warning(
                f"Layout algorithm '{layout}' failed: {e}. Using spring layout instead."
            )
            pos = nx.spring_layout(G, k=1, iterations=50)

        # Create the plot
        plt.figure(figsize=figsize)

        # Draw edges first (so nodes appear on top)
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=1.5, edge_color="gray")

        # Draw nodes with cluster colors
        unique_clusters = np.unique(cluster_colors)

        # Use default palette if no custom colors provided
        if custom_colors is None:
            custom_colors = get_palette_dict(len(unique_clusters))
            self.logger.debug(f"Using default color palette: {custom_colors}")

        # Handle custom colors
        if custom_colors is not None:
            if isinstance(custom_colors, dict):
                # Map cluster IDs to colors
                node_color_list = [
                    custom_colors.get(int(c), "gray") for c in cluster_colors
                ]
                self.logger.debug(
                    f"Using color dict, first 5 colors: {node_color_list[:5]}"
                )
            elif isinstance(custom_colors, list):
                # List of colors, one per cluster
                cluster_id_to_index = {
                    int(cid): idx for idx, cid in enumerate(unique_clusters)
                }
                node_color_list = [
                    custom_colors[cluster_id_to_index[int(c)]] for c in cluster_colors
                ]
            else:
                node_color_list = None

            if node_color_list:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    node_color=node_color_list,
                    node_size=node_size,
                    alpha=0.8,
                )
            else:
                # Fallback to default
                self._draw_nodes_default(
                    G, pos, cluster_colors, unique_clusters, node_size, cmap
                )
        else:
            # Use default colormap behavior
            self._draw_nodes_default(
                G, pos, cluster_colors, unique_clusters, node_size, cmap
            )

        # Optionally add labels for small graphs
        if G.number_of_nodes() <= 50:
            nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title(f"Graph Structure with Clusters (layout: {layout})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def _draw_nodes_default(
        self, G, pos, cluster_colors_list, unique_clusters, node_size, cmap
    ):
        """Default node drawing with color palette or colormap."""
        # Use color palette by default, unless cmap is explicitly provided
        if cmap is None:
            # Use default color palette
            palette_dict = get_palette_dict(len(unique_clusters))
            node_color_list = [
                palette_dict.get(int(c), "gray") for c in cluster_colors_list
            ]
            nx.draw_networkx_nodes(
                G,
                pos,
                node_color=node_color_list,
                node_size=node_size,
                alpha=0.8,
            )
        else:
            # Use colormap if explicitly provided
            # Ensure cluster IDs are properly normalized for the colormap
            if len(unique_clusters) > 1:
                # Normalize cluster IDs to [0, 1] for proper colormap mapping
                cluster_colors_normalized = [
                    (c - unique_clusters.min())
                    / (unique_clusters.max() - unique_clusters.min())
                    if unique_clusters.max() > unique_clusters.min()
                    else 0.0
                    for c in cluster_colors_list
                ]
            else:
                cluster_colors_normalized = cluster_colors_list

            # Use provided cmap
            colormap = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

            nx.draw_networkx_nodes(
                G,
                pos,
                node_color=cluster_colors_normalized,
                node_size=node_size,
                cmap=colormap,
                alpha=0.8,
                vmin=0,
                vmax=1,
            )

    def plot_fiedler_vector(
        self,
        output_dir: Path,
        filename: str = "fiedler_vector.png",
        figsize: tuple[int, int] = (10, 6),
    ):
        """
        Plot the sorted Fiedler vector values as a line plot.

        Args:
            output_dir: Directory where the plot will be saved
            filename: Name of the output file
            figsize: Figure size (width, height)
        """
        # Get the Fiedler vector (this will compute it if not already computed)
        fiedler_vec = self.cluster_machine.fiedler_vector

        # Create a figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot sorted Fiedler vector values
        sorted_values = np.sort(fiedler_vec)
        ax.plot(sorted_values, "b-", linewidth=1.5)
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Node Index (sorted by Fiedler vector value)")
        ax.set_ylabel("Fiedler Vector Value")
        ax.set_title("Sorted Fiedler Vector Values")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_fiedler_vector_on_graph(
        self,
        output_dir: Path,
        filename: str = "fiedler_vector_graph.png",
        layout: str = "spring",
        node_size: int = 100,
        figsize: tuple[int, int] = (12, 10),
        max_nodes: int = 500,
    ):
        """
        Plot the Fiedler vector values on the graph structure, with nodes colored
        by their Fiedler vector value. This helps visualize how the Fiedler vector
        partitions the graph.

        Args:
            output_dir: Directory where the plot will be saved
            filename: Name of the output file
            layout: Layout algorithm to use ('spring', 'kamada_kawai', 'circular', 'spectral')
            node_size: Size of nodes in the plot
            figsize: Figure size (width, height)
            max_nodes: Maximum number of nodes to plot (for large graphs)
        """
        # Get the Fiedler vector
        fiedler_vec = self.cluster_machine.fiedler_vector

        # Convert sparse matrix to NetworkX graph
        graph_matrix = self.cluster_machine.graph
        G = nx.from_scipy_sparse_array(graph_matrix)

        # Optionally sample nodes for large graphs
        if G.number_of_nodes() > max_nodes:
            self.logger.warning(
                f"Graph has {G.number_of_nodes()} nodes, sampling {max_nodes} nodes for visualization"
            )
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[
                :max_nodes
            ]
            sampled_nodes = [node for node, _ in top_nodes]
            G = G.subgraph(sampled_nodes).copy()
            fiedler_values = [fiedler_vec[node] for node in G.nodes()]
        else:
            fiedler_values = [fiedler_vec[node] for node in G.nodes()]

        # Choose layout algorithm
        layout_funcs = {
            "spring": nx.spring_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
            "spectral": nx.spectral_layout,
        }

        if layout not in layout_funcs:
            self.logger.warning(f"Unknown layout '{layout}', using 'spring' instead")
            layout = "spring"

        try:
            if layout == "kamada_kawai" and G.number_of_nodes() > 100:
                self.logger.warning(
                    "Kamada-Kawai layout is slow for large graphs, using spring layout instead"
                )
                pos = nx.spring_layout(G, k=1, iterations=50)
            else:
                pos = layout_funcs[layout](G)
        except Exception as e:
            self.logger.warning(
                f"Layout algorithm '{layout}' failed: {e}. Using spring layout instead."
            )
            pos = nx.spring_layout(G, k=1, iterations=50)

        # Create the plot
        plt.figure(figsize=figsize)

        # Draw edges first
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=1.5, edge_color="gray")

        # Draw nodes colored by Fiedler vector values
        # Use a diverging colormap (e.g., RdBu_r) to show positive/negative values
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=fiedler_values,
            node_size=node_size,
            cmap=plt.cm.RdBu_r,
            alpha=0.8,
            vmin=min(fiedler_values),
            vmax=max(fiedler_values),
        )

        # Add colorbar
        plt.colorbar(nodes, label="Fiedler Vector Value", shrink=0.8)

        # Optionally add labels for small graphs
        if G.number_of_nodes() <= 50:
            nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title(f"Fiedler Vector on Graph Structure (layout: {layout})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
