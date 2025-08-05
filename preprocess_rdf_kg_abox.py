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
        elif p in object_properties:
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

    # Function to ensure all entities and relations are in the training set
    def ensure_all_entities_and_relations_in_train(
        df, all_entities, all_relations, max_samples_per_item=3
    ):
        train_set = pd.DataFrame(columns=df.columns)
        used_indices = set()

        # Ensure all entities are included
        for entity in all_entities:
            entity_triples = df[(df["subject"] == entity) | (df["object"] == entity)]
            samples = entity_triples.sample(
                min(len(entity_triples), max_samples_per_item), random_state=42
            )
            used_indices.update(samples.index)
            train_set = pd.concat([train_set, samples], ignore_index=True)

        # Ensure all relations are included
        for relation in all_relations:
            if not (train_set["predicate"] == relation).any():
                relation_triples = df[df["predicate"] == relation]
                samples = relation_triples.sample(
                    min(len(relation_triples), max_samples_per_item), random_state=42
                )
                used_indices.update(samples.index)
                train_set = pd.concat([train_set, samples], ignore_index=True)

        train_set.drop_duplicates(inplace=True)
        return train_set, used_indices

    # Ensure all entities and relations are in the training set
    train_set, used_indices = ensure_all_entities_and_relations_in_train(
        df_shuffled, all_entities, all_relations
    )

    # Remove the selected triples from the shuffled DataFrame accurately
    train_keys = train_set[["subject", "predicate", "object"]].apply(tuple, axis=1)
    df_keys = df_shuffled[["subject", "predicate", "object"]].apply(tuple, axis=1)
    remaining_df = df_shuffled[~df_keys.isin(train_keys)].reset_index(drop=True)

    # Determine how many more triples to add to training set
    target_train_frac = 0.7
    n_total = len(df_shuffled)
    n_target_train = int(target_train_frac * n_total)
    n_to_add = n_target_train - len(train_set)

    # Add more triples to train set (stratified by predicate)
    if n_to_add > 0:
        stratify_col = remaining_df["predicate"]
        add_to_train_df, remaining_df = train_test_split(
            remaining_df, train_size=n_to_add, stratify=stratify_col, random_state=42
        )
        df_train = pd.concat([train_set, add_to_train_df], ignore_index=True)
    else:
        df_train = train_set

    df_val, df_test = train_test_split(
        remaining_df, test_size=0.5, stratify=remaining_df["predicate"], random_state=42
    )

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
        root = root
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
    train_ratio, val_ratio = 0.8, 0.1  # Remaining 0.1 is test

    #  Stratified sampling by predicate
    for pred, group in df_numeric.groupby("predicate"):
        group = group.sample(frac=1, random_state=42)  # Shuffle
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.append(group.iloc[:n_train])
        val.append(group.iloc[n_train : n_train + n_val])
        test.append(group.iloc[n_train + n_val :])

    #  Concatenate results
    df_train = pd.concat(train).reset_index(drop=True)
    df_val = pd.concat(val).reset_index(drop=True)
    df_test = pd.concat(test).reset_index(drop=True)

    #  Save
    if store:
        os.makedirs(root, exist_ok=True)
        literal_root = f"{root}/literals"
        os.makedirs(literal_root, exist_ok=True)
        df_train.to_csv(
            f"{literal_root}/train.txt", sep="\t", header=False, index=False
        )
        df_test.to_csv(f"{literal_root}/test.txt", sep="\t", header=False, index=False)
        df_val.to_csv(f"{literal_root}/valid.txt", sep="\t", header=False, index=False)
    print(
        "Enhanced train, validation, and test splits  for numeric literals created and saved."
    )


def rdf_to_lp(dataset, path=None):
    if path is None:
        kg_path = f"RDF_KGs/{dataset}.owl"
        target_dir = f"{dataset}_abox"
    else:
        kg_path = path
        target_dir = path.split("/")[-1].split(".")[0]
    datatype_df, object_df = get_data_obj_triples_split(kg_path)
    split_obj_property_df(object_df, root=target_dir, store=True)
    split_data_property_df(datatype_df, root=target_dir, store=True)


if __name__ == "__main__":
    rdf_to_lp(dataset="mutagenesis")
