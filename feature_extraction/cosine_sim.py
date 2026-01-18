#!/usr/bin/env python3
import numpy as np
import sys

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} vec1.npy vec2.npy")
        sys.exit(2)

    v1 = np.load(sys.argv[1])
    v2 = np.load(sys.argv[2])

    print(f"dim(v1)={v1.size} dim(v2)={v2.size}")
    print(f"cosine_similarity={cosine_sim(v1, v2):.6f}")

if __name__ == "__main__":
    main()
