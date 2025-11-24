import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EigenPlotter:
    def __init__(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "EigenPlotter initialized with %d eigenvalues", len(eigenvalues)
        )

    def plot_sorted_eigenvalues(
        self,
        output_dir: Path,
        filename: str = "sorted_eigenvalues.png",
        title="Histogram of Sorted Eigenvalues",
    ):
        """
        Plot eigenvalues where:
        - X-axis: eigenvalue index (position in sorted array)
        - Y-axis: eigenvalue value
        """
        sorted_eigs = np.sort(self.eigenvalues)
        max_eig = np.max(self.eigenvalues)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bars with unit width (no gaps)
        x_positions = np.arange(len(sorted_eigs))

        # Create alternating shades of blue for better legibility
        # Light blue and darker blue alternating
        light_blue = (0.5, 0.7, 0.9)  # Light blue RGB
        dark_blue = (0.2, 0.4, 0.8)  # Darker blue RGB
        colors = [
            light_blue if i % 2 == 0 else dark_blue for i in range(len(sorted_eigs))
        ]

        ax.bar(
            x_positions,
            sorted_eigs,
            width=1.0,  # Unit width ensures no gaps
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue Value")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        # Add max eigenvalue text at bottom
        plt.text(
            0.5,
            0.02,
            f"Max eigenvalue: {max_eig:.4f}",
            transform=fig.transFigure,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()
        self.logger.info("Saved sorted eigenvalues plot to %s", output_dir / filename)

    def plot_eigenvector(
        self, index: int, output_dir: Path, filename: str = None, title=None
    ):
        """
        Plot the values of a specific eigenvector and save to a file.
        """
        if index < 0 or index >= self.eigenvectors.shape[1]:
            self.logger.error("Eigenvector index %d out of bounds", index)
            return

        vec = self.eigenvectors[:, index]
        plt.figure(figsize=(8, 5))
        plt.plot(vec, marker="o")
        plt.xlabel("Node index")
        plt.ylabel("Eigenvector value")
        plt.title(title or f"Eigenvector {index} (λ={self.eigenvalues[index]:.4f})")
        plt.grid(True)
        out_file = filename or f"eigenvector_{index}.png"
        plt.savefig(output_dir / out_file)
        plt.close()
        self.logger.info("Saved eigenvector plot to %s", output_dir / out_file)

    def plot_eigenvector_heatmap(
        self, index: int, grid_shape, output_dir: Path, filename: str = None, title=None
    ):
        """
        Plot a heatmap of a 2D grid graph eigenvector and save to a file.

        Args:
            index: eigenvector index
            grid_shape: tuple (rows, cols) of the 2D grid
            output_dir: directory where plot will be saved
            filename: name of the output file
        """
        if index < 0 or index >= self.eigenvectors.shape[1]:
            self.logger.error("Eigenvector index %d out of bounds", index)
            return

        vec = self.eigenvectors[:, index]
        if vec.size != grid_shape[0] * grid_shape[1]:
            self.logger.error(
                "Eigenvector size %d does not match grid shape %s", vec.size, grid_shape
            )
            return

        heatmap = vec.reshape(grid_shape)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(heatmap, cmap="viridis", origin="lower")
        plt.colorbar(im, label="Eigenvector value")
        plt.title(
            title or f"Eigenvector {index} Heatmap (λ={self.eigenvalues[index]:.4f})"
        )
        plt.xlabel("Column")
        plt.ylabel("Row")
        out_file = filename or f"eigenvector_{index}_heatmap.png"
        plt.savefig(output_dir / out_file)
        plt.close()
        self.logger.info("Saved heatmap plot to %s", output_dir / out_file)
