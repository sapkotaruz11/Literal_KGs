import rdflib
import pandas as pd
from rdflib import RDF, RDFS, OWL, Literal
from rdflib.namespace import split_uri
from sklearn.model_selection import train_test_split
import os

RDF = rdflib.namespace.RDF
OWL = rdflib.namespace.OWL


# ---  Helper to return full URI (namespace + local name) ---
def get_local_name(term):
    try:
        ns, local = split_uri(term)
        return ns + local
    except:
        return str(term)


def is_from_ontology_ns(p):
    # ---  Check if predicate is from a known ontology namespace ---
    ontology_namespaces = {RDF, RDFS, OWL}
    return any(str(p).startswith(str(ns)) for ns in ontology_namespaces)


def get_data_obj_triples_split(path: str):
    graph = rdflib.Graph()
    graph.parse(path, format="xml")

    # --- Identify datatype and object properties ---
    datatype_properties = {
        p for p, _, _ in graph.triples((None, RDF.type, OWL.DatatypeProperty))
    }
    object_properties = {
        p for p, _, _ in graph.triples((None, RDF.type, OWL.ObjectProperty))
    }

    # --- Lists to hold triples ---
    datatype_triples = []
    object_triples = []

    # ---  Iterate through graph and classify triples ---
    for s, p, o in graph:
        if p in datatype_properties:
            datatype_triples.append(
                (
                    get_local_name(s),
                    get_local_name(p),
                    o.value if isinstance(o, Literal) else get_local_name(o),
                )
            )
        elif p in object_properties or is_from_ontology_ns(p):
            object_triples.append(
                (get_local_name(s), get_local_name(p), get_local_name(o))
            )

    # --- Create DataFrames ---
    datatype_df = pd.DataFrame(
        datatype_triples, columns=["subject", "predicate", "object"]
    )
    object_df = pd.DataFrame(object_triples, columns=["subject", "predicate", "object"])

    return datatype_df, object_df


def split_obj_property_df(df, root, store=False):
    # Extract unique entities and relations
    all_entities = set(df["subject"]).union(set(df["object"]))
    all_relations = set(df["predicate"])

    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Initialize splits
    train_list, test_list, val_list = [], [], []
    
    # For each predicate, ensure proper stratification
    for predicate in all_relations:
        pred_df = df_shuffled[df_shuffled["predicate"] == predicate].copy()
        n_total = len(pred_df)
        
        # Calculate split sizes (60-30-10)
        n_train = max(1, int(n_total * 0.6))  # Ensure at least 1 in train
        n_test = max(1, int(n_total * 0.3)) if n_total > 1 else 0
        n_val = n_total - n_train - n_test
        
        # Ensure we don't exceed total
        if n_train + n_test + n_val > n_total:
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        
        # Split the predicate data
        train_list.append(pred_df.iloc[:n_train])
        if n_test > 0:
            test_list.append(pred_df.iloc[n_train:n_train + n_test])
        if n_val > 0:
            val_list.append(pred_df.iloc[n_train + n_test:n_train + n_test + n_val])
    
    # Combine all splits
    df_train = pd.concat(train_list, ignore_index=True)
    df_test = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=df.columns)
    df_val = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame(columns=df.columns)
    
    # Ensure all entities appear in training set (add missing entities)
    train_entities = set(df_train["subject"]).union(set(df_train["object"]))
    missing_entities = all_entities - train_entities
    
    if missing_entities:
        print(f"Adding {len(missing_entities)} missing entities to training set")
        for entity in missing_entities:
            # Find triples containing this entity in test or val
            entity_in_test = df_test[(df_test["subject"] == entity) | (df_test["object"] == entity)]
            entity_in_val = df_val[(df_val["subject"] == entity) | (df_val["object"] == entity)]
            
            # Move one triple from test or val to train
            if not entity_in_test.empty:
                move_triple = entity_in_test.iloc[0:1]
                df_train = pd.concat([df_train, move_triple], ignore_index=True)
                df_test = df_test.drop(entity_in_test.index[0]).reset_index(drop=True)
            elif not entity_in_val.empty:
                move_triple = entity_in_val.iloc[0:1]
                df_train = pd.concat([df_train, move_triple], ignore_index=True)
                df_val = df_val.drop(entity_in_val.index[0]).reset_index(drop=True)

    # Function to log stats of each split
    def log_stats(df, name):
        unique_entities = set(df["subject"]).union(set(df["object"]))
        unique_relations = set(df["predicate"])
        print(
            f"{name} - Triples: {len(df)}, Entities: {len(unique_entities)}, Relations: {len(unique_relations)}"
        )

    log_stats(df_train, "Train")
    log_stats(df_val, "Validation")
    log_stats(df_test, "Test")

    if store:
        os.makedirs(root, exist_ok=True)
        df_train.to_csv(f"{root}/train.txt", sep="\t", header=False, index=False)
        df_test.to_csv(f"{root}/test.txt", sep="\t", header=False, index=False)
        df_val.to_csv(f"{root}/valid.txt", sep="\t", header=False, index=False)
    print("Enhanced train, validation, and test splits created and saved.")


def split_data_property_df(df, root, store=False):
    def is_strictly_numeric(value):
        try:
            v = float(value)
            return not isinstance(value, bool)
        except (ValueError, TypeError):
            return False

    #  Filter only strictly numeric triples
    df_numeric = df[df["object"].apply(is_strictly_numeric)].copy()
    # Step 2: Initialize splits
    train, val, test = [], [], []
    train_ratio, test_ratio, val_ratio = 0.6, 0.3, 0.1  # 60-30-10 split

    #  Stratified sampling by predicate
    for pred, group in df_numeric.groupby("predicate"):
        group = group.sample(frac=1, random_state=42)  # Shuffle
        n = len(group)
        n_train = int(n * train_ratio)
        n_test = int(n * test_ratio)

        train.append(group.iloc[:n_train])
        test.append(group.iloc[n_train : n_train + n_test])
        val.append(group.iloc[n_train + n_test :])  # Remaining goes to validation

    #  Concatenate results
    df_train = pd.concat(train).reset_index(drop=True)
    df_val = pd.concat(val).reset_index(drop=True)
    df_test = pd.concat(test).reset_index(drop=True)

    #  Save
    if store:
        os.makedirs(root, exist_ok=True)
        df_train.to_csv(
            f"{root}/literals/train.txt", sep="\t", header=False, index=False
        )
        df_test.to_csv(f"{root}/literals/test.txt", sep="\t", header=False, index=False)
        df_val.to_csv(f"{root}/literals/valid.txt", sep="\t", header=False, index=False)
    print(
        "Enhanced train, validation, and test splits  for numeric literals created and saved."
    )


def rdf_to_lp(dataset, path=None):
    if path is None:
        kg_path = f"RDF_KGs/{dataset}.owl"
        target_dir = dataset
    else:
        kg_path = path
        target_dir = path.split("/")[-1].split(".")[0]
    datatype_df, object_df = get_data_obj_triples_split(kg_path)
    split_obj_property_df(object_df, root=target_dir, store=True)
    split_data_property_df(datatype_df, root=target_dir, store=True)


if __name__ == "__main__":
    rdf_to_lp(dataset="mutagenesis")
