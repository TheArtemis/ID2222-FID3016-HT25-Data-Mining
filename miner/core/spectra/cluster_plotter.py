import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

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
