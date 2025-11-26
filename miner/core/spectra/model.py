from pydantic import BaseModel
import numpy as np


class EighResult(BaseModel):
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return (
            f"Spectra(eigenvalues={self.eigenvalues}, eigenvectors={self.eigenvectors})"
        )

    def __str__(self):
        return (
            f"Spectra(eigenvalues={self.eigenvalues}, eigenvectors={self.eigenvectors})"
        )


class ClusterAnalysisResult(BaseModel):
    data: dict[str, float]

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        trimmed_data = {k: f"{v:.2f}" for k, v in self.data.items()}
        return f"{trimmed_data}"

    def __str__(self):
        trimmed_data = {k: f"{v:.2f}" for k, v in self.data.items()}
        return f"{trimmed_data}"
