import torch
from typing import List, Set, Tuple, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class HexCoordinate:
    """Axial coordinates for hexagonal grid positioning."""
    q: int
    r: int

    def __hash__(self):
        return hash((self.q, self.r))

    def __eq__(self, other):
        return isinstance(other, HexCoordinate) and self.q == other.q and self.r == other.r

    def __repr__(self):
        return f"H({self.q},{self.r})"

    def neighbors(self) -> List['HexCoordinate']:
        """Return the 6 direct neighbors in hexagonal grid."""
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        return [HexCoordinate(self.q + dq, self.r + dr) for dq, dr in directions]

    def distance(self, other: 'HexCoordinate') -> int:
        """Hexagonal grid distance."""
        return (abs(self.q - other.q) + abs(self.q + self.r - other.q - other.r) +
                abs(self.r - other.r)) // 2


def generate_hex_grid(radius: int) -> List[HexCoordinate]:
    """Generate hexagonal grid coordinates within given radius."""
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append(HexCoordinate(q, r))
    return coords


def build_adjacency_matrix(coords: List[HexCoordinate]) -> torch.Tensor:
    """Build adjacency matrix for hexagonal grid topology."""
    n = len(coords)
    adj = torch.zeros(n, n)
    coord_to_idx = {coord: i for i, coord in enumerate(coords)}

    for i, coord in enumerate(coords):
        for neighbor in coord.neighbors():
            if neighbor in coord_to_idx:
                j = coord_to_idx[neighbor]
                adj[i, j] = 1.0

    return adj
