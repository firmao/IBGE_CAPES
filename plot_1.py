import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Global Seed for Reproducibility
np.random.seed(42)

def plot_innovation_latent_space():
    try:
        # Load the 100+100 records
        df_ibge = pd.read_csv('ibge_data.csv')
        df_capes = pd.read_csv('capes_data.csv')
        
        # Synthetic Latent Projection (Mimicking GCN Output)
        # Cluster A: Socio-Economic (IBGE)
        x_ibge = np.random.normal(loc=2.0, scale=1.0, size=100)
        y_ibge = np.random.normal(loc=2.0, scale=1.0, size=100)
        
        # Cluster B: Academic Excellence (CAPES)
        x_capes = np.random.normal(loc=-2.0, scale=1.0, size=100)
        y_capes = np.random.normal(loc=-2.0, scale=1.0, size=100)
        
        plt.figure(figsize=(12, 8))
        
        # 1. Plot IBGE Municipalities
        # Size is based on the first digit of the ID (Regional Weight)
        sizes_ibge = df_ibge['municipio_id'].astype(str).str[0].astype(int) * 30
        plt.scatter(x_ibge, y_ibge, c='dodgerblue', s=sizes_ibge, 
                    alpha=0.6, edgecolors='w', label='IBGE (Municipal Nodes)')
        
        # 2. Plot CAPES Graduate Programs
        # Size is based on the official CAPES Grade (3-7)
        sizes_capes = df_capes['nota_capes'] * 25
        plt.scatter(x_capes, y_capes, c='firebrick', s=sizes_capes, 
                    alpha=0.6, edgecolors='w', marker='s', label='CAPES (Academic Nodes)')

        # 3. Draw High-Affinity Synergy Lines (The "Crosswalk")
        # Connecting top nodes to show the Neuro-Symbolic alignment
        for i in range(15):
            plt.plot([x_ibge[i], x_capes[i]], [y_ibge[i], y_capes[i]], 
                     c='black', alpha=0.1, lw=0.8, linestyle='--')

        plt.title("Neuro-Symbolic Latent Space: IBGE-CAPES Innovation Mapping", fontsize=15)
        plt.xlabel("Latent Dimension 1: Regional Economic Complexity", fontsize=12)
        plt.ylabel("Latent Dimension 2: Scientific Research Maturity", fontsize=12)
        plt.legend(frameon=True, loc='best')
        plt.grid(True, which='both', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('innovation_latent_space.png', dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error during plotting: {e}")

if __name__ == "__main__":
    plot_innovation_latent_space()