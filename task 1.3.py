import numpy as np
from collections import deque

def grado(A):
    return A.sum(axis=1)

def clustering(A):
    k = grado(A)
    tri = np.diag(A @ A @ A)          # 2 * triángulos por nodo
    C = np.zeros(len(A), dtype=float)
    m = k > 1
    C[m] = tri[m] / (k[m] * (k[m] - 1))
    return C

def distancia_promedio(A):
    n = len(A)
    total, pares = 0, 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in np.nonzero(A[u])[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for t in range(s + 1, n):
            if dist[t] > 0:           # solo pares con camino
                total += dist[t]
                pares += 1
    return total / pares if pares else 0.0

A = np.array([[0,1,1,0,0],
              [1,0,1,1,0],
              [1,1,0,0,1],
              [0,1,0,0,1],
              [0,0,1,1,0]])

print(grado(A))               # [2 3 3 2 2]
print(clustering(A))          # [1. 0.333 0.333 0. 0.]
print(distancia_promedio(A))  # 1.4