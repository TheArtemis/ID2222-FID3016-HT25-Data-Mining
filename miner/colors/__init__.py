"""Color palette module for visualizations."""

# Default color palette for cluster visualizations
DEFAULT_PALETTE = [
    "#124e78",  # Yale Blue
    "#f2bb05",  # Amber Gold
    "#d74e09",  # Spicy Orange
    "#95a3a4",  # Cool Steel
    "#6e0e0a",  # Dark Garnet
]

# Extended palette (for more than 5 clusters, cycles through)
EXTENDED_PALETTE = DEFAULT_PALETTE + [
    "#c9ada7",  # Light gray-purple
    "#9a8c98",  # Medium gray-purple
    "#4a4e69",  # Dark blue-gray
    "#22223b",  # Very dark blue
]


def get_palette(num_colors: int) -> list[str]:
    """
    Get color palette for a given number of colors.

    Args:
        num_colors: Number of colors needed

    Returns:
        List of color hex codes
    """
    if num_colors <= len(DEFAULT_PALETTE):
        return DEFAULT_PALETTE[:num_colors]
    else:
        # Cycle through extended palette if needed
        colors = []
        for i in range(num_colors):
            colors.append(EXTENDED_PALETTE[i % len(EXTENDED_PALETTE)])
        return colors


def get_palette_dict(num_colors: int) -> dict[int, str]:
    """
    Get color dictionary mapping indices to colors.

    Args:
        num_colors: Number of colors needed

    Returns:
        Dictionary mapping index (0, 1, 2, ...) to color hex code
    """
    colors = get_palette(num_colors)
    return {i: colors[i] for i in range(num_colors)}
