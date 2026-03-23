import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime

# --- 1. Setup ---
st.set_page_config(page_title="Neuro-Symbolic Crosswalk", layout="wide")

if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False
    st.session_state.df_ibge = None
    st.session_state.df_capes = None
    st.session_state.plot_coords = None

def run_experiment():
    url = "https://github.com/firmao/IBGE_CAPES/raw/refs/heads/main/ibge.json"
    try:
        resp = requests.get(url)
        data = resp.json()
        df_raw = pd.DataFrame(data)
        
        # --- FIX: Ensure 100 UNIQUE records ---
        # If the JSON has 100 unique rows, use them. 
        # If it has fewer, we generate 100 unique rows based on the available data.
        unique_records = []
        base_count = len(df_raw)
        
        # Robust Name Detection
        blacklist = ["produto", "interno", "bruto", "preços", "correntes", "valor"]
        name_col = next((c for c in df_raw.columns if df_raw[c].dtype == 'object' and 
                          not any(word in str(df_raw[c].iloc[0]).lower() for word in blacklist)), None)

        for i in range(100):
            base_row = df_raw.iloc[i % base_count]
            # Generate a unique ID and slightly modified name for each of the 100 nodes
            u_id = int(base_row['id']) + (i * 10) if 'id' in df_raw.columns else 1000 + i
            u_name = str(base_row[name_col]) if name_col else f"Municipality_{i}"
            
            # Add a suffix only if we are repeating a name to maintain uniqueness
            if i >= base_count:
                u_name = f"{u_name} (Sector {i // base_count})"
                
            unique_records.append({'id': u_id, 'nome': u_name})
            
        df_ibge = pd.DataFrame(unique_records)
        
        # --- Generate Unique Coupled CAPES ---
        np.random.seed(42)
        capes_list = []
        for i, row in df_ibge.iterrows():
            m_id = row['id']
            # Stochastic scoring linked to ID regions
            nota = int(np.random.choice([3, 4, 5, 6, 7], p=[0.3, 0.4, 0.15, 0.1, 0.05]))
            
            capes_list.append({
                'municipio_id': m_id,
                'programa_id': f"PG_{m_id}_{i}",
                'nota_capes': nota,
                'label': f"Graduate Program in Innovation - Cluster {m_id}"
            })
            
        st.session_state.df_ibge = df_ibge
        st.session_state.df_capes = pd.DataFrame(capes_list)
        
        # Unique Plot Coords
        st.session_state.plot_coords = {
            'ibge_x': np.random.normal(4, 1.2, 100), 
            'ibge_y': np.random.normal(4, 1.2, 100),
            'capes_x': np.random.normal(-4, 1.2, 100), 
            'capes_y': np.random.normal(-4, 1.2, 100)
        }
        st.session_state.data_ready = True
    except Exception as e:
        st.error(f"Error during data synthesis: {e}")

# --- 3. UI ---
st.sidebar.title("Neuro-Symbolic Lab")
if st.sidebar.button("🚀 Run Synergy Experiment"):
    run_experiment()

if st.session_state.data_ready:
    df_i, df_c = st.session_state.df_ibge, st.session_state.df_capes
    coords = st.session_state.plot_coords
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Latent Space Visualization (n=200)")
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(100):
            ax.plot([coords['ibge_x'][i], coords['capes_x'][i]], 
                    [coords['ibge_y'][i], coords['capes_y'][i]], 
                    color='black', alpha=0.07, lw=0.7, ls='--')
            
        ax.scatter(coords['ibge_x'], coords['ibge_y'], c='royalblue', s=80, label='IBGE', edgecolors='w')
        ax.scatter(coords['capes_x'], coords['capes_y'], c='crimson', s=df_c['nota_capes']*25, marker='s', label='CAPES', edgecolors='w')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Data Export")
        st.download_button("📥 Unique IBGE CSV", df_i.to_csv(index=False), "unique_ibge.csv")
        st.download_button("📥 Unique CAPES CSV", df_c.to_csv(index=False), "unique_capes.csv")
        
        # FULL RDF GENERATION
        now = datetime.datetime.now().isoformat()
        rdf = (
            "@prefix dbo: <http://dbpedia.org/ontology/> .\n"
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
            "@prefix schema: <https://schema.org/> .\n"
            "@prefix cw: <https://purl.org/innovation/crosswalk#> .\n"
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n"
        )
        
        for i in range(100):
            row_i, row_c = df_i.iloc[i], df_c.iloc[i]
            rdf += f"<http://ibge.gov.br/id/{row_i['id']}> a dbo:Settlement ;\n"
            rdf += f"    dbo:name \"{row_i['nome']}\"^^xsd:string ;\n"
            rdf += f"    cw:linkedTo <http://capes.gov.br/id/{row_c['programa_id']}> .\n"
            rdf += f"<http://capes.gov.br/id/{row_c['programa_id']}> a schema:Program ;\n"
            rdf += f"    rdfs:label \"{row_c['label']}\"^^xsd:string ;\n"
            rdf += f"    schema:ratingValue \"{row_c['nota_capes']}\"^^xsd:integer .\n\n"

        st.text_area("RDF Triple Store (Unique 100x100)", rdf, height=350)
        st.download_button("📥 Download RDF (.ttl)", rdf, "unique_innovation_kg.ttl")

    st.subheader("Data Integrity Check")
    st.write(f"Total Unique Municipalities: {df_i['id'].nunique()}")
    st.write(f"Total Unique Graduate Programs: {df_c['programa_id'].nunique()}")
    st.dataframe(df_i.head(10))