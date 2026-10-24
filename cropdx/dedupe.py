"""Near-duplicate detection that is invariant to flips / 90-degree rotations.

PlantVillage copies on Kaggle often contain flipped/rotated versions of the same leaf photo.
If one copy lands in train and another in test the test score is inflated (data leakage).

Method: 64-bit difference-hash of a 9x9 grayscale thumbnail, computed for all 8 dihedral
variants. Candidate pairs are found with pigeon-hole chunk buckets (exact for Hamming <= max),
then VERIFIED with the mean absolute pixel difference of 32x32 thumbnails.
"""
import numpy as np
from PIL import Image

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    return ((x * _H01) >> np.uint64(56)).astype(np.int32)


def fingerprint(gray: Image.Image):
    """-> (32x32 uint8 thumbnail, 8 uint64 hashes, variant v = 2*rot + flip)."""
    thumb = np.asarray(gray.resize((32, 32), Image.BOX), dtype=np.uint8)
    small = np.asarray(gray.resize((9, 9), Image.BOX), dtype=np.float32)
    hashes = []
    for k in range(4):
        r = np.rot90(small, k)
        for v in (r, r[:, ::-1]):
            bits = v[:8, 1:] > v[:8, :-1]
            hashes.append(int.from_bytes(np.packbits(bits.ravel()).tobytes(), "big"))
    return thumb, np.array(hashes, dtype=np.uint64)


def _variant_batch(t, v):
    k, f = divmod(int(v), 2)
    r = np.rot90(t, k, axes=(1, 2))
    return r[:, :, ::-1] if f else r


def near_pairs(hashes, thumbs, max_hamming=6):
    """Return arrays (a, b, mad): image a matches (a variant of) image b with verified distance mad."""
    base = hashes[:, 0].copy()
    allh = hashes.reshape(-1)
    nch = max_hamming + 1
    bounds = np.linspace(0, 64, nch + 1).astype(int)
    found = []
    for c in range(nch):
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        mask, sh = np.uint64((1 << (hi - lo)) - 1), np.uint64(lo)
        kb, ka = (base >> sh) & mask, (allh >> sh) & mask
        ob, oa = np.argsort(kb, kind="stable"), np.argsort(ka, kind="stable")
        kbs, kas = kb[ob], ka[oa]
        ukeys, starts, counts = np.unique(kbs, return_index=True, return_counts=True)
        left, right = np.searchsorted(kas, ukeys, "left"), np.searchsorted(kas, ukeys, "right")
        for u in range(len(ukeys)):
            if counts[u] == 1 and right[u] - left[u] == 1:
                continue
            bi = ob[starts[u]:starts[u] + counts[u]]
            ei = oa[left[u]:right[u]]
            eh = allh[ei]
            for s in range(0, len(bi), 256):
                bc = bi[s:s + 256]
                d = popcount64(base[bc][:, None] ^ eh[None, :])
                rr, cc = np.nonzero(d <= max_hamming)
                if rr.size:
                    own, ent = bc[rr], ei[cc]
                    keep = (ent // 8) != own
                    found.append(np.stack([own[keep], ent[keep]], 1))
    if not found:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32)
    pairs = np.unique(np.concatenate(found).astype(np.int64), axis=0)
    a, b, v = pairs[:, 0], pairs[:, 1] // 8, pairs[:, 1] % 8
    mad = np.empty(len(a), np.float32)
    for vv in range(8):
        idx = np.nonzero(v == vv)[0]
        for s in range(0, len(idx), 20000):
            ii = idx[s:s + 20000]
            ta = thumbs[a[ii]].astype(np.int16)
            tb = _variant_batch(thumbs[b[ii]], vv).astype(np.int16)
            mad[ii] = np.abs(ta - tb).mean(axis=(1, 2))
    return a, b, mad


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)
