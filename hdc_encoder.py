import numpy as np
from typing import List, Tuple

class HDCEncoder:
    """A simple Hyperdimensional Computing encoder for 2D trajectories.
    
    - Uses bipolar hypervectors (-1, +1) of dimension `dim`.
    - Quantizes x,y to a grid and maps each grid coordinate to random HVs.
    - Binds x and y HVs elementwise and circularly shifts by timestep to
      encode temporal order, then sums and bipolarizes.
    """
    
    def __init__(self, dim=10000, x_bins=64, y_bins=48, seed=42):
        self.dim = dim
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.rng = np.random.RandomState(seed)
        # item memory for x and y bins
        self.x_im = self._rand_hvs(self.x_bins)
        self.y_im = self._rand_hvs(self.y_bins)
    
    def _rand_hvs(self, n):
        # bipolar HVs
        return self.rng.choice([-1.0, 1.0], size=(n, self.dim)).astype(np.float32)
    
    def _quantize(self, x: int, y: int, w: int, h: int) -> Tuple[int, int]:
        # map pixel coords to bin indices
        bx = int((x / float(max(1, w))) * (self.x_bins - 1))
        by = int((y / float(max(1, h))) * (self.y_bins - 1))
        bx = max(0, min(self.x_bins - 1, bx))
        by = max(0, min(self.y_bins - 1, by))
        return bx, by
    
    def _cshift(self, hv: np.ndarray, k: int) -> np.ndarray:
        # circular shift (temporal binding via permutation)
        return np.roll(hv, k)
    
    def encode_sequence(self, points: List[Tuple[int,int]], frame_size: Tuple[int,int]) -> np.ndarray:
        """Encode a list of (x,y) points into a single HV.
        
        points: list of (x,y) in pixel coordinates.
        frame_size: (width, height)
        """
        if not points:
            return np.zeros(self.dim, dtype=np.float32)
        w, h = frame_size
        acc = np.zeros(self.dim, dtype=np.float32)
        for t, (x,y) in enumerate(points):
            bx, by = self._quantize(x, y, w, h)
            xhv = self.x_im[bx]
            yhv = self.y_im[by]
            bound = xhv * yhv  # elementwise multiply (binding)
            perm = self._cshift(bound, t % self.dim)
            acc += perm
        # bipolarize by sign
        acc = np.where(acc >= 0, 1.0, -1.0)
        return acc
    
    def bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        # superpose then bipolarize
        if not hvs:
            return np.zeros(self.dim, dtype=np.float32)
        s = np.sum(hvs, axis=0)
        return np.where(s >= 0, 1.0, -1.0)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # cosine similarity for bipolar vectors
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class AssociativeMemory:
    def __init__(self):
        self.prototypes = {}  # label -> hv (np.ndarray)
    
    def add(self, label: str, hv: np.ndarray):
        if label in self.prototypes:
            # average with existing
            self.prototypes[label] = np.where(self.prototypes[label] + hv >= 0, 1.0, -1.0)
        else:
            self.prototypes[label] = hv.copy()
    
    def query(self, hv: np.ndarray, topk: int = 1):
        if not self.prototypes:
            return []
        sims = [(lab, float(np.dot(hv, p) / (np.linalg.norm(hv) * np.linalg.norm(p) + 1e-9)))
                for lab, p in self.prototypes.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:topk]
