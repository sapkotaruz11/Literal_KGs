import os
import pandas as pd


def count_entities_relations_and_splits(dataset_path):
    # File paths for train, test, and valid (assuming the files are .txt and tab-separated)
    train_file = os.path.join(dataset_path, "train.txt")
    test_file = os.path.join(dataset_path, "test.txt")
    valid_file = os.path.join(dataset_path, "valid.txt")

    # Load the data files (using \t as the separator since it's tab-separated)
    train_data = pd.read_csv(
        train_file, sep="\t", header=None, names=["head", "relation", "tail"]
    )
    test_data = pd.read_csv(
        test_file, sep="\t", header=None, names=["head", "relation", "tail"]
    )
    valid_data = pd.read_csv(
        valid_file, sep="\t", header=None, names=["head", "relation", "tail"]
    )

    # Combine all data to extract unique entities and relations
    all_data = pd.concat([train_data, test_data, valid_data])

    # Count unique entities and relations
    unique_entities = set(all_data["head"]).union(set(all_data["tail"]))
    unique_relations = set(all_data["relation"])

    # Count entries in each split
    train_count = len(train_data)
    test_count = len(test_data)
    valid_count = len(valid_data)

    # Return the results as a dictionary
    return {
        "dataset": dataset_path,
        "total_entities": len(unique_entities),
        "total_relations": len(unique_relations),
        "train_count": train_count,
        "test_count": test_count,
        "valid_count": valid_count,
    }


def count_literal_entities_relations_and_splits(dataset_path):
    # Path to the literals folder
    literals_folder = os.path.join(dataset_path, "literals")

    # File paths for train, test, and valid in literals (using .txt format)
    train_file = os.path.join(literals_folder, "train.txt")
    test_file = os.path.join(literals_folder, "test.txt")
    valid_file = os.path.join(literals_folder, "valid.txt")

    # Load the literal data files
    train_data = pd.read_csv(
        train_file, sep="\t", header=None, names=["head", "relation", "tail"]
    )
    test_data = pd.read_csv(
        test_file, sep="\t", header=None, names=["head", "relation", "tail"]
    )
    valid_data = pd.read_csv(
        valid_file, sep="\t", header=None, names=["head", "relation", "tail"]
    )

    # Combine all data to extract unique entities and relations
    all_data = pd.concat([train_data, test_data, valid_data])
    # Count unique entities and relations
    unique_entities = set(all_data["head"])
    unique_relations = set(all_data["relation"])

    # Count entries in each split
    train_count = len(train_data)
    test_count = len(test_data)
    valid_count = len(valid_data)

    # Return the results as a dictionary
    return {
        "dataset": dataset_path + "_literals",
        "total_entities": len(unique_entities),
        "total_relations": len(unique_relations),
        "train_count": train_count,
        "test_count": test_count,
        "valid_count": valid_count,
    }


def analyse_dataset(root: str = None, datasets=None, store=False):
    if root is None:
        root = "KGs/"
    if datasets is None:
        datasets = ["DB15k", "FB15k-237", "mutagenesis", "YAGO15k"]
    results = []
    # List of datasets to process
    for dataset in datasets:
        dataset_path = os.path.join(root, dataset)
        # Create an empty list to store results
        result = count_entities_relations_and_splits(dataset_path)
        results.append(result)

    for dataset in datasets:
        dataset_path = os.path.join(root, dataset)
        literals_result = count_literal_entities_relations_and_splits(dataset)
        results.append(literals_result)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(results)
    results_df.to_csv("dataset_statistics.csv", sep="\t", index=False, header=True)
    return results_df


if __name__ == "__main__":
    # Print the DataFrame
    print(analyse_dataset(root=""))
