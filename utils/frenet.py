import numpy as np
from dataclasses import dataclass

@dataclass
class FrenetFrame:
    """
    Frenet frame sobre un centerline 2D.
    centerline: array de shape (N, 2) con puntos [x, y] en orden de recorrido.
    """
    centerline: np.ndarray      # (N, 2)
    s_accum: np.ndarray         # (N,) distancia acumulada
    seg_vecs: np.ndarray        # (N-1, 2) vectores de cada segmento
    seg_lengths: np.ndarray     # (N-1,)
    normals: np.ndarray         # (N-1, 2) normales unitarias

    @classmethod
    def from_centerline(cls, centerline: np.ndarray) -> "FrenetFrame":
        centerline = np.asarray(centerline, dtype=float)
        if centerline.ndim != 2 or centerline.shape[1] != 2:
            raise ValueError("centerline debe ser de shape (N, 2)")

        seg_vecs = np.diff(centerline, axis=0)              # v_i = p_{i+1} - p_i
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        # evitar divisiones por cero
        seg_lengths_safe = seg_lengths + 1e-12

        # distancia acumulada s
        s_accum = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        # normales unitarias (rotación 90°: [-vy, vx])
        normals = np.empty_like(seg_vecs)
        normals[:, 0] = -seg_vecs[:, 1]
        normals[:, 1] = seg_vecs[:, 0]
        norms_n = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        normals /= norms_n

        return cls(
            centerline=centerline,
            s_accum=s_accum,
            seg_vecs=seg_vecs,
            seg_lengths=seg_lengths_safe,
            normals=normals,
        )

    def xy_to_sd(self, point: np.ndarray):
        """
        Convierte un punto (x, y) a coordenadas Frenet (s, d).
        s: distancia a lo largo del centerline
        d: offset lateral (positivo hacia la normal)
        """
        p = np.asarray(point, dtype=float)
        cl = self.centerline
        v = self.seg_vecs
        s_acc = self.s_accum
        L = self.seg_lengths
        n = self.normals

        best_dist2 = np.inf
        best_s = 0.0
        best_d = 0.0

        # recorremos segmentos
        for i in range(len(v)):
            p0 = cl[i]
            vi = v[i]

            w = p - p0
            denom = np.dot(vi, vi)
            if denom < 1e-16:
                continue

            # proyección escalar t en [0,1]
            t = np.dot(w, vi) / denom
            t = np.clip(t, 0.0, 1.0)

            proj = p0 + t * vi
            diff = p - proj
            dist2 = np.dot(diff, diff)

            if dist2 < best_dist2:
                best_dist2 = dist2
                # coordenada s: s_i + t * |vi|
                best_s = s_acc[i] + t * L[i]
                # coordenada d: proyección sobre normal
                best_d = np.dot(diff, n[i])

        return best_s, best_d

    def sd_to_xy(self, s: float, d: float):
        """
        Convierte coordenadas Frenet (s, d) de vuelta a (x, y).
        s se clampa dentro [0, s_max].
        """
        cl = self.centerline
        v = self.seg_vecs
        n = self.normals
        s_acc = self.s_accum
        L = self.seg_lengths

        # clamp s al rango del path
        s = float(s)
        d = float(d)
        if s <= 0.0:
            base = cl[0]
            seg_idx = 0
            t = 0.0
        elif s >= s_acc[-1]:
            base = cl[-1]
            seg_idx = len(v) - 1
            t = 1.0
        else:
            # encontrar segmento donde cae s
            seg_idx = np.searchsorted(s_acc, s) - 1
            seg_idx = int(np.clip(seg_idx, 0, len(v) - 1))
            ds = s - s_acc[seg_idx]
            t = ds / L[seg_idx]
            t = np.clip(t, 0.0, 1.0)
            base = cl[seg_idx] + t * v[seg_idx]

        # desplazamiento lateral
        xy = base + d * n[seg_idx]
        return xy
