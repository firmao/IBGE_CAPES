import pandas as pd
import requests
from unicodedata import normalize

# --- 1. FETCH OFFICIAL IBGE DATA ---
def get_official_ibge():
    print("Fetching official IBGE municipality list...")
    url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
    response = requests.get(url)
    if response.status_code == 200:
        ibge_data = response.json()
        # Create a reference mapping: { 'CLEANED_NAME': '7_DIGIT_ID' }
        return {normalize_text(m['nome']): m['id'] for m in ibge_data}
    else:
        raise Exception("Could not reach IBGE API.")

def normalize_text(text):
    if not text: return ""
    # Standardizing: remove accents, uppercase, and trim
    return normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

# --- 2. FETCH OFFICIAL CAPES DATA ---
def get_capes_data():
    print("Connecting to CAPES Open Data Portal...")
    # NOTE: This is the official URL pattern for CAPES datasets. 
    # If the file for the specific year changes, update the link below.
    capes_url = "https://dadosabertos.capes.gov.br/dataset/30140220-413e-4251-8178-005118742886/resource/666c5a27-a06f-473d-9860-848e6583d735/download/br-capes-ppg-2023.csv"
    
    # We read directly from the portal; using a sample for performance in this proof
    try:
        df = pd.read_csv(capes_url, sep=';', encoding='latin-1', nrows=5000)
        return df
    except Exception as e:
        print(f"Error accessing CAPES: {e}")
        return None

# --- 3. THE EXPERIMENT ---
def analyze_quality():
    # Load sources
    ibge_reference = get_official_ibge()
    df_capes = get_capes_data()
    
    if df_capes is not None:
        # Step A: Identify the location column in CAPES (usually 'NM_MUNICIPIO_PROGRAMA_IES')
        # We normalize the CAPES city names to match the IBGE format
        df_capes['normalized_city'] = df_capes['NM_MUNICIPIO_PROGRAMA_IES'].apply(normalize_text)
        
        # Step B: Attempt the Linkage
        df_capes['ibge_id'] = df_capes['normalized_city'].map(ibge_reference)
        
        # Step C: Quality Metrics
        linked_count = df_capes['ibge_id'].notna().sum()
        total_records = len(df_capes)
        accuracy_rate = (linked_count / total_records) * 100
        
        print("\n--- EXPERIMENT RESULTS ---")
        print(f"Total CAPES Records Analyzed: {total_records}")
        print(f"Successfully Linked to IBGE: {linked_count}")
        print(f"Quality Score (Completeness): {accuracy_rate:.2f}%")
        
        # Output results to a CSV for your paper's graph generation
        df_capes[['NM_MUNICIPIO_PROGRAMA_IES', 'ibge_id']].to_csv("quality_results.csv", index=False)

if __name__ == "__main__":
    analyze_quality()
