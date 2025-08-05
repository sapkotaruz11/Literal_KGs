import rdflib
import pandas as pd
from rdflib import RDF, RDFS, OWL, Literal
from rdflib.namespace import split_uri
from collections import Counter
import os

def analyze_rdf_kg_structure(kg_path):
    """Analyze the structure of the RDF knowledge graph"""
    
    print(f"=== Analyzing RDF Knowledge Graph: {kg_path} ===")
    
    # Load the graph
    graph = rdflib.Graph()
    graph.parse(kg_path, format="xml")
    
    print(f"Total triples in graph: {len(graph)}")
    
    # Identify different types of properties
    datatype_properties = {
        p for p, _, _ in graph.triples((None, RDF.type, OWL.DatatypeProperty))
    }
    object_properties = {
        p for p, _, _ in graph.triples((None, RDF.type, OWL.ObjectProperty))
    }
    
    print(f"Declared datatype properties: {len(datatype_properties)}")
    print(f"Declared object properties: {len(object_properties)}")
    
    # Analyze all predicates used in the graph
    all_predicates = set()
    predicate_counts = Counter()
    datatype_triples = []
    object_triples = []
    other_triples = []
    
    for s, p, o in graph:
        all_predicates.add(p)
        predicate_counts[p] += 1
        
        # Classify triples
        if p in datatype_properties:
            datatype_triples.append((s, p, o))
        elif p in object_properties:
            object_triples.append((s, p, o))
        elif str(p).startswith(str(RDF)) or str(p).startswith(str(RDFS)) or str(p).startswith(str(OWL)):
            object_triples.append((s, p, o))  # Ontology predicates treated as object properties
        else:
            other_triples.append((s, p, o))
    
    print(f"\nTotal unique predicates: {len(all_predicates)}")
    print(f"Datatype triples: {len(datatype_triples)}")
    print(f"Object triples: {len(object_triples)}")
    print(f"Other triples: {len(other_triples)}")
    
    print(f"\n--- Predicate Usage Statistics ---")
    print(f"{'Predicate':<50} {'Count':<8} {'Type':<15}")
    print("-" * 75)
    
    for pred, count in predicate_counts.most_common():
        pred_short = str(pred).split('/')[-1] if '/' in str(pred) else str(pred).split('#')[-1]
        if pred in datatype_properties:
            pred_type = "Datatype"
        elif pred in object_properties:
            pred_type = "Object"
        elif str(pred).startswith(str(RDF)) or str(pred).startswith(str(RDFS)) or str(pred).startswith(str(OWL)):
            pred_type = "Ontology"
        else:
            pred_type = "Unknown"
        
        print(f"{pred_short:<50} {count:<8} {pred_type:<15}")
    
    # Analyze literal values for datatype properties
    if datatype_triples:
        print(f"\n--- Datatype Property Analysis ---")
        for pred in datatype_properties:
            pred_triples = [t for t in datatype_triples if t[1] == pred]
            if pred_triples:
                values = [t[2] for t in pred_triples]
                numeric_values = []
                string_values = []
                
                for val in values:
                    if isinstance(val, Literal):
                        try:
                            float(val.value)
                            numeric_values.append(val.value)
                        except (ValueError, TypeError):
                            string_values.append(val.value)
                    else:
                        try:
                            float(val)
                            numeric_values.append(val)
                        except (ValueError, TypeError):
                            string_values.append(val)
                
                pred_short = str(pred).split('/')[-1] if '/' in str(pred) else str(pred).split('#')[-1]
                print(f"{pred_short}: {len(pred_triples)} triples, {len(numeric_values)} numeric, {len(string_values)} string")
    
    # Analyze entity coverage
    entities = set()
    for s, p, o in graph:
        if not isinstance(o, Literal):  # Subject and non-literal objects are entities
            entities.add(s)
            entities.add(o)
        else:
            entities.add(s)  # Only subjects for literal triples
    
    print(f"\nTotal unique entities: {len(entities)}")
    
    return {
        'total_triples': len(graph),
        'datatype_properties': len(datatype_properties),
        'object_properties': len(object_properties),
        'datatype_triples': len(datatype_triples),
        'object_triples': len(object_triples),
        'entities': len(entities),
        'predicates': len(all_predicates)
    }

if __name__ == "__main__":
    # Analyze the mutagenesis dataset
    analyze_rdf_kg_structure("RDF_KGs/mutagenesis.owl")
