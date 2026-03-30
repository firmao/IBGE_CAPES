import numpy as np

def simulate_regional_impact():
    # Node 0: Anápolis (IBGE - Transport Sector)
    # Node 1: Anápolis (CAPES - MSc Production Engineering)
    # Node 2: São Paulo (CAPES - Theoretical Physics)
    
    # Features: [Regional_Focus, Industrial_Link, Technical_Depth]
    X = np.array([
        [0.9, 0.8, 0.2], # IBGE Transport (Needs logistics info)
        [0.95, 0.9, 0.7], # CAPES MSc Eng (Strong local logistics focus)
        [0.1, 0.2, 0.95]  # CAPES Physics (High depth, no local logistics link)
    ])

    # Adjacency Matrix (A): Node 0 and 1 are in the same city
    A = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])

    # GCN Normalization: D^-0.5 * A * D^-0.5
    D = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
    A_hat = D @ A @ D

    # GCN Forward Pass (Simplified Weight Matrix)
    W = np.eye(3)
    Z = np.maximum(0, A_hat @ X @ W)

    # Compute Similarity (The Crosswalk Discovery)
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_local = cosine_sim(Z[0], Z[1]) # Transport <-> MSc Eng
    sim_metro = cosine_sim(Z[0], Z[2]) # Transport <-> Physics

    print(f"--- GCN 'Invisible Insight' Discovery ---")
    print(f"Confidence (Anápolis MSc -> Local Transport): {sim_local:.4f}")
    print(f"Confidence (Anápolis Transport -> SP Physics): {sim_metro:.4f}")

if __name__ == "__main__":
    simulate_regional_impact()
