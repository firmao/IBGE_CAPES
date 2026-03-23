# 🌐 Hybrid Neuro-Symbolic Crosswalk: IBGE-CAPES
This repository implements a Hybrid Neuro-Symbolic Crosswalk to identify latent synergies between Brazilian socio-economic infrastructure (**IBGE**) and academic research excellence (**CAPES**). By combining Graph Convolutional Network (GCN) logic with Semantic Web technologies, this tool maps regional innovation ecosystems into a machine-readable Knowledge Graph.

## 🚀 Key Features
* **Latent Manifold Alignment:** Projects heterogeneous data (100 Municipalities, 100 Graduate Programs) into a unified vector space.

* **FAIR-Compliant Output:** Automatically serializes results into RDF/Turtle format using DBpedia (dbo) and Schema.org vocabularies.

* **Symbolic Provenance:** Integration of W3C PROV ontology to track data lineage and transformation.

* **Interactive Lab:** A Streamlit-based interface for real-time experiment execution and data visualization.


## 📊 Architecture Overview

The HNSA operates in three distinct phases:

1.  Mapping RDF-triples from [IBGE SIDRA](https://www.google.com/search?q=https://sidra.ibge.gov.br/) and [CAPES Open Data](https://dadosabertos.capes.gov.br/).
2.  Injecting verified subgraphs into the LLM prompt to ensure 100% data traceability.

## 📑 Methodology: 
The Neuro-Symbolic Approach: 
The system operates in two distinct phases:
1. The Neural Layer (Sub-symbolic): A simulated GCN architecture processes socio-economic tensors to identify "hidden" affinities between cities and research hubs.
2. The Symbolic Layer (Knowledge Representation): These affinities are reified into explicit RDF triples.
We apply FAIR Principles to ensure the output is:
- Findable: Unique URIs for every municipality and program.
- Accessible: Open Turtle (.ttl) exports for triple-store integration.
- Interoperable: Use of standard ontologies ($xsd, rdfs, dbo, schema$).
- Reusable: Strict literal typing (e.g., xsd:integer for scores) for precise SPARQL querying.

-----
## 🔍 SPARQL Query Example
Once you download the innovation_kg.ttl file, you can perform complex audits such as finding high-performing ecosystems:

```
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX cw: <https://purl.org/innovation/crosswalk#>
PREFIX schema: <https://schema.org/>

SELECT ?cityName ?score
WHERE {
  ?city dbo:name ?cityName ;
        cw:linkedTo ?program .
  ?program schema:ratingValue ?score .
  FILTER(?score >= 5)
}
ORDER BY DESC(?score)
```

## 🛠️ Installation

This version is designed to be "plug-and-play" on any Python environment (Windows/Linux/Mac) without driver issues.

```bash
# Clone the repository
git clone https://github.com/firmao/IBGE_CAPES.git
cd IBGE_CAPES

# Install dependencies
pip install -r requirements.txt
```

-----

## 💻 Usage

To run the full suite (GCN link discovery, visualization):

web interface (Preferably, with more information and more complete):
```bash
streamlit run .\web.py
```

- only LLM and plot (not updated).
```bash
python allexperiments.py
```

### Expected Output

<img width="1808" height="919" alt="image" src="https://github.com/user-attachments/assets/d5cde520-6454-44f8-8c1c-73944b3473ca" />

**Visualization:** A Matplotlib window showing the semantic alignment of IBGE municipalities and CAPES research centers.
-----
## 📚 Data Provenance

  * **IBGE:** Brazilian Institute of Geography and Statistics ([sidra.ibge.gov.br](https://www.google.com/search?q=https://sidra.ibge.gov.br/)).
  * **CAPES:** Coordination for the Improvement of Higher Education Personnel ([dadosabertos.capes.gov.br](https://dadosabertos.capes.gov.br/)).

-----

## 📄 Citation

If you use this architecture in your research, please cite:

```bibtex
@article{Valdestilhas2026hnsa,
  title={Hybrid Neuro-Symbolic Crosswalk: Mapping Regional Innovation Ecosystems via Latent Manifold Alignment and Semantic Web Integration},
  author={Valdestilhas, et al.},
  journal={TBA},
  year={2026},
  url={https://github.com/firmao/IBGE_CAPES}
}
```

# Scientific Justification (For my records)
- **The "Silo" Problem**: We are identifying that the data exists, but isn't linked.

- **The "Hallucination" Problem**: We are pinpointing the specific failure of LLMs (probabilistic vs. deterministic).

- **The "Neuro-Symbolic" Solution**: We are positioning my HNSA as the bridge between these two worlds.
