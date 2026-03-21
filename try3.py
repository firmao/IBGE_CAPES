import pandas as pd
import requests
from rdflib import Graph, Literal, RDF, URIRef, Namespace, OWL
from rdflib.namespace import XSD, FOAF

# --- 1. CONFIGURATION ---
BR = Namespace("http://gov.br/ontology/")
SCHEMA_ORG = Namespace("http://schema.org/")
# Manually define the SCHEMA namespace to avoid the ImportError
SCHEMA = Namespace("https://schema.org/")

def fetch_real_ibge_gdp(muni_id):
    """
    Fetches real GDP data using the IBGE SIDRA API (Table 5938 - PIB at current prices).
    Direct Link: https://api.sidra.ibge.gov.br/v1/values/t/5938/n6/all/v/37/p/last/f/c
    """
    # Specific query for the individual municipality ID
    url = f"https://api.sidra.ibge.gov.br/v1/values/t/5938/n6/{muni_id}/v/37/p/last/f/c"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        # SIDRA returns a list where index 0 is the header and index 1 is the value
        if len(data) > 1:
            return float(data[1]['V']) 
        return 0.0
    except Exception as e:
        print(f"Error fetching data for {muni_id}: {e}")
        return 0.0

def run_corrected_experiment():
    g = Graph()
    g.bind("br", BR)
    g.bind("schema", SCHEMA_ORG)

    # Official IDs (IBGE 7-digit codes)
    # 3550308 = Sao Paulo | 3304557 = Rio de Janeiro | 3106200 = Belo Horizonte
    cities = [
        {"id": "3550308", "name": "Sao_Paulo", "capes_score": 7},
        {"id": "3304557", "name": "Rio_de_Janeiro", "capes_score": 6},
        {"id": "3106200", "name": "Belo_Horizonte", "capes_score": 6}
    ]

    print("Step 1: Fetching LIVE IBGE GDP via SIDRA API...")

    for city in cities:
        muni_uri = URIRef(f"http://gov.br/muni/{city['id']}")
        gdp_val = fetch_real_ibge_gdp(city['id'])
        
        # RDF Triple Generation
        g.add((muni_uri, RDF.type, SCHEMA_ORG.City))
        g.add((muni_uri, SCHEMA_ORG.name, Literal(city['name'])))
        g.add((muni_uri, BR.academicScore, Literal(city['capes_score'], datatype=XSD.integer)))
        g.add((muni_uri, BR.totalGDP, Literal(gdp_val, datatype=XSD.float)))
        
        print(f"Captured: {city['name']} | GDP: R$ {gdp_val:,.2f}")

    # Step 2: SPARQL Analysis
    print("\n--- KNOWLEDGE GRAPH ANALYSIS ---")
    query = """
    PREFIX br: <http://gov.br/ontology/>
    SELECT ?name ?score ?gdp
    WHERE {
        ?c br:academicScore ?score ;
           <http://schema.org/name> ?name ;
           br:totalGDP ?gdp .
    }
    """
    for row in g.query(query):
        print(f"Muni: {row.name} | Score: {row.score} | GDP: {row.gdp}")

if __name__ == "__main__":
    run_corrected_experiment()