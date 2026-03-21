import requests
from rdflib import Graph , Literal , RDF , URIRef , Namespace
from rdflib . namespace import FOAF , DCTERMS
# Define Namespaces
IBGE = Namespace ( " https :// servicodados . ibge . gov . br / api / v1 / bngb / " )
CAPES = Namespace ( " https :// periodicos . capes . gov . br / mcp / " )
GAO = Namespace ( " http :// example . org / gao # " )
def create_knowledge_graph ( target_city ) :
g = Graph ()
# 1. Fetch from IBGE BNGB API
# URL : https :// servicodados . ibge . gov . br / api / v1 / bngb /
nomegeografico /
resp = requests . get ( f " { IBGE } nomegeografico / " , params ={ ’ nome ’:
target_city })
if resp . status_code == 200 and resp . json () :
data = resp . json () [0]
feature_id = data . get ( ’ id ’ , ’ 001 ’)
feature_uri = URIRef ( f " { IBGE } feature /{ feature_id } " )
# Add Geographic metadata to Graph
g . add (( feature_uri , RDF . type , GAO . GeographicFeature ) )
g . add (( feature_uri , FOAF . name , Literal ( data . get ( ’
nome_geografico ’) ) ) )
# 2. Fetch from CAPES ( Simulated logic for MCP search )
# Using https :// www . npmjs . com / package / periodicos - capes - mcp
logic
article_uri = URIRef ( f " { CAPES } article /10.1000/ example_doi " )
# 3. Create the Linked Data Bridge
g . add (( feature_uri , GAO . isSubjectOfScientificStudy ,
article_uri ) )
g . add (( article_uri , DCTERMS . title , Literal ( f " Scientific
Analysis of { target_city } " ) ) )
return g
# Execute and serialize in Turtle format
kg = create_knowledge_graph ( " Brasilia " )
print ( kg . serialize ( format = " turtle " ) )