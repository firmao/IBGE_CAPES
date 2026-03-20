import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from unicodedata import normalize
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_robust_session():
    session = requests.Session()
    # Retry strategy: 3 attempts, backoff factor helps avoid spamming the server
    retries = Retry(total=3, backoff_factor=1, status_forcelcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

# --- STAGE 1: FETCH OFFICIAL IBGE MUNICIPALITY LIST ---
def get_ibge_reference():
    url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
    session = get_robust_session()
    
    try:
        # Added an explicit timeout of 10 seconds
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {normalize_text(m['nome']): m['id'] for m in data}
    except requests.exceptions.RequestException as e:
        print(f"Critical Error: {e}")
        # For the paper, we return an empty dict or a cached local fallback
        return {}

def normalize_text(text):
    """Standardizes strings: Upper case, no accents, no trailing spaces."""
    if not text or pd.isna(text): return ""
    return normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

# --- STAGE 2: FETCH OFFICIAL CAPES GRADUATE DATA ---
def get_capes_dataset():
    """Pulls the 'Programas de Pós-Graduação' dataset from CAPES Open Data."""
    print("Accessing CAPES Portal...")
    # This is the 2024/2025 persistent link for the Graduate Programs dataset
    capes_url = "https://dadosabertos.capes.gov.br/dataset/30140220-413e-4251-8178-005118742886/resource/666c5a27-a06f-473d-9860-848e6583d735/download/br-capes-ppg-2023.csv"
    
    try:
        response = requests.get(capes_url, timeout=30)
        # Using a buffer to handle the latin-1 encoding common in BR Gov data
        df = pd.read_csv(io.StringIO(response.content.decode('latin-1')), sep=';')
        return df.head(5000) # Analyzing 5,000 records as per paper requirement
    except Exception as e:
        print(f"Error: {e}. Ensure you have an active internet connection.")
        return None

# --- STAGE 3: THE QUALITY ANALYSIS ---
def run_experiment():
    ibge_map = get_ibge_reference()
    df_capes = get_capes_dataset()
    
    if df_capes is not None:
        # 1. Direct Linkage (Raw)
        df_capes['raw_match'] = df_capes['NM_MUNICIPIO_PROGRAMA_IES'].map(ibge_map)
        
        # 2. Optimized Linkage (Normalized)
        df_capes['norm_city'] = df_capes['NM_MUNICIPIO_PROGRAMA_IES'].apply(normalize_text)
        df_capes['ibge_id'] = df_capes['norm_city'].map(ibge_map)
        
        # Calculate Metrics
        raw_success = (df_capes['raw_match'].notna().sum() / 5000) * 100
        norm_success = (df_capes['ibge_id'].notna().sum() / 5000) * 100
        
        # --- STAGE 4: GENERATE ACADEMIC CHARTS ---
        plt.figure(figsize=(12, 6))
        
        # Chart: Success Rate Comparison
        plt.subplot(1, 2, 1)
        sns.barplot(x=['Raw Data', 'Normalized Data'], y=[raw_success, norm_success], palette='viridis')
        plt.title('Comparison: Linkage Success Rate')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        
        # Chart: Record Distribution
        plt.subplot(1, 2, 2)
        status_counts = df_capes['ibge_id'].notna().value_counts()
        plt.pie(status_counts, labels=['Linked', 'Orphan'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
        plt.title('Final Data Quality Distribution')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nQuality Result: {norm_success:.2f}% of records successfully linked using official IDs.")

if __name__ == "__main__":
    run_experiment()