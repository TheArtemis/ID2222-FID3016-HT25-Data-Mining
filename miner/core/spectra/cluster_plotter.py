import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
import networkx as nx

from miner.core.spectra.cluster_machine import ClusterMachine


class ClusterPlotter:
    """Class for plotting clustering results and eigenvalue analysis."""

    def __init__(self, cluster_machine: ClusterMachine):
        """
        Initialize the plotter with a ClusterMachine instance.

        Args:
            cluster_machine: The ClusterMachine instance containing the data to plot
        """
        self.cluster_machine = cluster_machine
        self.logger = logging.getLogger(__name__)

    def plot_clusters(self, output_dir: Path, filename: str = "clusters.png"):
        """
        Plot the clusters in 2D space using the first two dimensions of Y.

        Args:
            output_dir: Directory where the plot will be saved
            filename: Name of the output file
        """
        if self.cluster_machine.latest_clusters is None:
            self.logger.error(
                "Before plotting, you need to cluster the data first: run .cluster() first"
            )
            return

        # Plot the clusters
        plt.scatter(
            self.cluster_machine.Y()[:, 0],
            self.cluster_machine.Y()[:, 1],
            c=self.cluster_machine.latest_clusters,
        )
        plt.savefig(output_dir / filename)
        plt.close()

    def plot_eigenvalue_spectrum(
        self, output_dir: Path, filename: str = "eigenvalues.png"
    ):
        """
        Plot eigenvalues from paper's Laplacian to identify number of clusters.

        Args:
            output_dir: Directory where the plot will be saved
            filename: Name of the output file
        """
        if self.cluster_machine.eigenvalues is None:
            # Compute all eigenvalues for spectrum analysis
            if self.cluster_machine.laplacian is None:
                self.cluster_machine.build_laplacian()
            eigenvalues, _ = eigsh(
                self.cluster_machine.laplacian,
                k=min(100, self.cluster_machine.laplacian.shape[0] - 1),
                which="LA",
            )
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        else:
            eigenvalues = self.cluster_machine.eigenvalues

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(eigenvalues)), eigenvalues, "bo-")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.title("Eigenvalue Spectrum (L = D^{-1/2} A D^{-1/2})")
        plt.grid(True)
        plt.savefig(output_dir / filename)
        plt.close()

    def plot_eigenvector_values(
        self, eigenvector_idx: int, output_dir: Path, filename: str = None
    ):
        """
        Plot sorted values of a specific eigenvector.

        Args:
            eigenvector_idx: Index of the eigenvector to plot (0-based)
            output_dir: Directory where the plot will be saved
            filename: Name of the output file (defaults to eigenvector_{idx}_sorted.png)
        """
        if self.cluster_machine.eigenvectors is None:
            self.logger.error("Compute eigenvectors first")
            return

        if eigenvector_idx >= self.cluster_machine.eigenvectors.shape[1]:
            self.logger.error(
                f"Only {self.cluster_machine.eigenvectors.shape[1]} eigenvectors available"
            )
            return

        eigenvec = self.cluster_machine.eigenvectors[:, eigenvector_idx]
        sorted_values = np.sort(eigenvec)

        if filename is None:
            filename = f"eigenvector_{eigenvector_idx}_sorted.png"

        plt.figure(figsize=(12, 6))
        plt.plot(sorted_values, "b-", linewidth=1)
        plt.xlabel("Node Index (sorted by eigenvector value)")
        plt.ylabel(f"Eigenvector {eigenvector_idx + 1} Value")
        plt.title(f"Sorted Eigenvector {eigenvector_idx + 1} (from largest eigenvalue)")
        plt.grid(True)
        plt.savefig(output_dir / filename)
        plt.close()

    def plot_graph_structure(
        self,
        output_dir: Path,
        filename: str = "graph_structure.png",
        layout: str = "spring",
        node_size: int = 50,
        figsize: tuple[int, int] = (12, 10),
        max_nodes: int | None = None,
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
        """
        if self.cluster_machine.latest_clusters is None:
            self.logger.error(
                "Before plotting, you need to cluster the data first: run .cluster() first"
            )
            return

        # Convert sparse matrix to NetworkX graph
        graph_matrix = self.cluster_machine.graph
        G = nx.from_scipy_sparse_array(graph_matrix)

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
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, edge_color="gray")

        # Draw nodes with cluster colors
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=cluster_colors,
            node_size=node_size,
            cmap=plt.cm.tab10,
            alpha=0.8,
        )

        # Optionally add labels for small graphs
        if G.number_of_nodes() <= 50:
            nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title(f"Graph Structure with Clusters (layout: {layout})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_fiedler_vector(
        self,
        output_dir: Path,
        filename: str = "fiedler_vector.png",
        figsize: tuple[int, int] = (14, 6),
    ):
        """
        Plot the Fiedler vector in multiple ways:
        1. Sorted values of the Fiedler vector
        2. Fiedler vector values on the graph structure (if graph is not too large)

        Args:
            output_dir: Directory where the plot will be saved
            filename: Name of the output file
            figsize: Figure size (width, height)
        """
        # Get the Fiedler vector (this will compute it if not already computed)
        fiedler_vec = self.cluster_machine.fiedler_vector

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Sorted Fiedler vector values
        sorted_values = np.sort(fiedler_vec)
        axes[0].plot(sorted_values, "b-", linewidth=1.5)
        axes[0].axhline(y=0, color="r", linestyle="--", alpha=0.5, linewidth=1)
        axes[0].set_xlabel("Node Index (sorted by Fiedler vector value)")
        axes[0].set_ylabel("Fiedler Vector Value")
        axes[0].set_title("Sorted Fiedler Vector Values")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Fiedler vector values as a bar plot (sorted)
        axes[1].bar(range(len(sorted_values)), sorted_values, width=0.8, alpha=0.7)
        axes[1].axhline(y=0, color="r", linestyle="--", alpha=0.5, linewidth=1)
        axes[1].set_xlabel("Node Index (sorted by Fiedler vector value)")
        axes[1].set_ylabel("Fiedler Vector Value")
        axes[1].set_title("Fiedler Vector Values (Bar Plot)")
        axes[1].grid(True, alpha=0.3, axis="y")

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
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, edge_color="gray")

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
