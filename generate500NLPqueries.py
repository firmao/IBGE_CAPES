import csv
import random

def export_experimental_dataset(filename="query_dataset_200.csv"):
    domains = ["Social Sciences", "Humanities", "Materials Science"]
    topics = {
        "Social Sciences": ["Demographics", "Public Policy", "Urban Sociology", "Economic Trends", "Educational Equality"],
        "Humanities": ["Digital History", "Historical Linguistics", "Archaeology", "Renaissance Art", "Comparative Literature"],
        "Materials Science": ["Nanomaterials", "Superconductors", "Polymers", "Crystallography", "Metallurgy"]
    }
    templates = [
        "What are the persistent identifiers for {topic} datasets?",
        "List all SPARQL endpoints available for {topic} research.",
        "Show metadata schemas used in {topic} projects.",
        "How is data provenance handled for {topic} in the GCN?",
        "Find interoperable vocabularies related to {topic}.",
        "Retrieve the license information for {topic} data assets.",
        "Which institutions host the most reusable {topic} data?",
        "Identify cross-domain links between {topic} and other GCN nodes."
    ]

    dataset = []
    seen = set()

    while len(dataset) < 200:
        dom = random.choice(domains)
        top = random.choice(topics[dom])
        temp = random.choice(templates)
        query = temp.format(topic=top)
        
        if query not in seen:
            dataset.append([len(dataset)+1, dom, top, query])
            seen.add(query)

    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Domain", "Topic", "Natural_Language_Query"])
        writer.writerows(dataset)
    
    print(f"Dataset with 200 queries exported to {filename}")

if __name__ == "__main__":
    export_experimental_dataset()