import rdflib
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import FOAF, DCTERMS
import random
import os

def generate_gcn_triples(output_file="gcn_1m_triples.nt", total_target=1000000):
    """
    Generates a National Knowledge Graph (GCN) with 1.5 million triples.
    Uses N-Triples format for memory efficiency and speed.
    """
    g = Graph()
    
    # Define Namespaces
    GCN = Namespace("http://purl.org/gcn/ontology/")
    EX = Namespace("http://example.org/data/")
    # Manually define the SCHEMA namespace to avoid the ImportError
    SCHEMA = Namespace("https://schema.org/")

    domains = ["SocialSciences", "Humanities", "MaterialsScience"]
    
    print(f"Starting generation of {total_target} triples...")

    # We write directly to file in chunks to avoid RAM overflow
    with open(output_file, "wb") as f:
        count = 0
        
        # 1. Generate Domain & Quality Metadata (The "Symbolic Anchor")
        for domain in domains:
            domain_uri = EX[domain]
            f.write(f"<{domain_uri}> <{RDF.type}> <{GCN.ResearchDomain}> .\n".encode())
            f.write(f"<{domain_uri}> <{DCTERMS.title}> \"{domain} Repository\" .\n".encode())
            count += 2

        # 2. Generate the Bulk Data (1.5M Triples)
        # Each 'Entity' will have ~5 triples (Type, Title, Author, Domain, FAIR Score)
        entities_needed = (total_target - count) // 5
        
        for i in range(entities_needed):
            entity_id = f"item_{i}"
            subj = EX[entity_id]
            dom = random.choice(domains)
            
            # Triple 1: Type
            f.write(f"<{subj}> <{RDF.type}> <{SCHEMA.Dataset}> .\n".encode())
            # Triple 2: Title
            f.write(f"<{subj}> <{DCTERMS.title}> \"Scientific Dataset {i}\" .\n".encode())
            # Triple 3: Domain Link
            f.write(f"<{subj}> <{GCN.belongsTo}> <{EX[dom]}> .\n".encode())
            # Triple 4: Creator (FOAF)
            f.write(f"<{subj}> <{DCTERMS.creator}> \"Researcher_{random.randint(1, 5000)}\" .\n".encode())
            # Triple 5: FAIR Score (Wilkinson Principle Grounding)
            f.write(f"<{subj}> <{GCN.fairScore}> \"{random.uniform(0.7, 1.0):.2f}\" .\n".encode())
            
            count += 5
            
            if i % 100000 == 0 and i > 0:
                print(f"Progress: {count} triples generated...")

        print(f"Final Count: {count} triples saved to {output_file}")

# Execute the generator
if __name__ == "__main__":
    generate_gcn_triples()