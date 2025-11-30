import torch
from typing import List, Set, Tuple, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class HexCoordinate:
    """Represents axial coordinates for positioning on a hexagonal grid.

    This dataclass provides a convenient way to store and operate on hexagonal
    coordinates. It includes methods for finding neighbors and calculating
    distances on the grid. The axial coordinate system is used for its
    simplicity in these calculations.

    Attributes:
        q (int): The 'q' coordinate in the axial system.
        r (int): The 'r' coordinate in the axial system.
    """

    q: int
    r: int

    def __hash__(self):
        return hash((self.q, self.r))

    def __eq__(self, other):
        return (
            isinstance(other, HexCoordinate) and self.q == other.q and self.r == other.r
        )

    def __repr__(self):
        return f"H({self.q},{self.r})"

    def neighbors(self) -> List["HexCoordinate"]:
        """Returns the 6 direct neighbors of this coordinate on the grid.

        Returns:
            List[HexCoordinate]: A list of the six neighboring coordinates.
        """
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        return [HexCoordinate(self.q + dq, self.r + dr) for dq, dr in directions]

    def distance(self, other: "HexCoordinate") -> int:
        """Calculates the hexagonal grid distance to another coordinate.

        This is the shortest number of steps required to move from this
        coordinate to the other on the hex grid.

        Args:
            other (HexCoordinate): The coordinate to measure the distance to.

        Returns:
            int: The distance between the two coordinates.
        """
        return (
            abs(self.q - other.q)
            + abs(self.q + self.r - other.q - other.r)
            + abs(self.r - other.r)
        ) // 2


def generate_hex_grid(radius: int) -> List[HexCoordinate]:
    """Generates a list of `HexCoordinate`s for a grid of a given radius.

    The grid is centered at (0,0) and includes all coordinates within the
    specified hexagonal distance from the center.

    Args:
        radius (int): The radius of the hexagonal grid.

    Returns:
        List[HexCoordinate]: A list of all coordinates in the grid.
    """
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append(HexCoordinate(q, r))
    return coords


def build_adjacency_matrix(coords: List[HexCoordinate]) -> torch.Tensor:
    """Builds an adjacency matrix for a given list of hexagonal coordinates.

    The resulting matrix represents the connectivity of the graph where an
    edge exists between two coordinates if they are direct neighbors on the
    hexagonal grid.

    Args:
        coords (List[HexCoordinate]): The list of coordinates (nodes) in the
            graph.

    Returns:
        torch.Tensor: A dense adjacency matrix of shape [n_coords, n_coords].
    """
    n = len(coords)
    adj = torch.zeros(n, n)
    coord_to_idx = {coord: i for i, coord in enumerate(coords)}

    for i, coord in enumerate(coords):
        for neighbor in coord.neighbors():
            if neighbor in coord_to_idx:
                j = coord_to_idx[neighbor]
                adj[i, j] = 1.0

    return adj
