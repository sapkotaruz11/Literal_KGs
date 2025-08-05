import pandas as pd
import numpy as np
from collections import Counter
import os

def analyze_stratification(dataset_path, literals=False):
    """Analyze the stratification quality of train/test/val splits"""
    
    if literals:
        base_path = f"{dataset_path}/literals"
    else:
        base_path = dataset_path
        
    # Load data
    train_df = pd.read_csv(f"{base_path}/train.txt", sep="\t", header=None, names=["subject", "predicate", "object"])
    test_df = pd.read_csv(f"{base_path}/test.txt", sep="\t", header=None, names=["subject", "predicate", "object"])
    val_df = pd.read_csv(f"{base_path}/valid.txt", sep="\t", header=None, names=["subject", "predicate", "object"])
    
    total_triples = len(train_df) + len(test_df) + len(val_df)
    
    print(f"\n=== Analysis for {'Literals' if literals else 'Object Properties'} in {dataset_path} ===")
    print(f"Total triples: {total_triples}")
    print(f"Train: {len(train_df)} ({len(train_df)/total_triples*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/total_triples*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/total_triples*100:.1f}%)")
    
    # Analyze predicate distribution
    train_pred_counts = Counter(train_df["predicate"])
    test_pred_counts = Counter(test_df["predicate"])
    val_pred_counts = Counter(val_df["predicate"])
    
    print(f"\n--- Predicate Distribution Analysis ---")
    all_predicates = set(train_pred_counts.keys()) | set(test_pred_counts.keys()) | set(val_pred_counts.keys())
    
    predicate_analysis = []
    for pred in all_predicates:
        train_count = train_pred_counts.get(pred, 0)
        test_count = test_pred_counts.get(pred, 0)
        val_count = val_pred_counts.get(pred, 0)
        total_pred = train_count + test_count + val_count
        
        train_ratio = train_count / total_pred * 100 if total_pred > 0 else 0
        test_ratio = test_count / total_pred * 100 if total_pred > 0 else 0
        val_ratio = val_count / total_pred * 100 if total_pred > 0 else 0
        
        predicate_analysis.append({
            'predicate': pred.split('/')[-1] if '/' in pred else pred,
            'total': total_pred,
            'train': train_count,
            'test': test_count,
            'val': val_count,
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'val_ratio': val_ratio
        })
    
    # Sort by total count
    predicate_analysis.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"{'Predicate':<30} {'Total':<8} {'Train':<8} {'Test':<8} {'Val':<8} {'T%':<6} {'Te%':<6} {'V%':<6}")
    print("-" * 85)
    for analysis in predicate_analysis:
        print(f"{analysis['predicate']:<30} {analysis['total']:<8} {analysis['train']:<8} "
              f"{analysis['test']:<8} {analysis['val']:<8} {analysis['train_ratio']:<6.1f} "
              f"{analysis['test_ratio']:<6.1f} {analysis['val_ratio']:<6.1f}")
    
    # Check entity coverage
    if not literals:  # Only for object properties
        print(f"\n--- Entity Coverage Analysis ---")
        train_entities = set(train_df["subject"]) | set(train_df["object"])
        test_entities = set(test_df["subject"]) | set(test_df["object"])
        val_entities = set(val_df["subject"]) | set(val_df["object"])
        all_entities = train_entities | test_entities | val_entities
        
        print(f"Total unique entities: {len(all_entities)}")
        print(f"Entities in train: {len(train_entities)} ({len(train_entities)/len(all_entities)*100:.1f}%)")
        print(f"Entities in test: {len(test_entities)} ({len(test_entities)/len(all_entities)*100:.1f}%)")
        print(f"Entities in val: {len(val_entities)} ({len(val_entities)/len(all_entities)*100:.1f}%)")
        
        # Check overlap
        test_only = test_entities - train_entities
        val_only = val_entities - train_entities
        print(f"Entities only in test (not in train): {len(test_only)}")
        print(f"Entities only in val (not in train): {len(val_only)}")
    else:  # For literals, check subject coverage
        print(f"\n--- Subject Coverage Analysis (Literals) ---")
        train_subjects = set(train_df["subject"])
        test_subjects = set(test_df["subject"])
        val_subjects = set(val_df["subject"])
        all_subjects = train_subjects | test_subjects | val_subjects
        
        print(f"Total unique subjects: {len(all_subjects)}")
        print(f"Subjects in train: {len(train_subjects)} ({len(train_subjects)/len(all_subjects)*100:.1f}%)")
        print(f"Subjects in test: {len(test_subjects)} ({len(test_subjects)/len(all_subjects)*100:.1f}%)")
        print(f"Subjects in val: {len(val_subjects)} ({len(val_subjects)/len(all_subjects)*100:.1f}%)")
        
        # Check overlap
        test_only = test_subjects - train_subjects
        val_only = val_subjects - train_subjects
        print(f"Subjects only in test (not in train): {len(test_only)}")
        print(f"Subjects only in val (not in train): {len(val_only)}")

if __name__ == "__main__":
    # Analyze the mutagenesis dataset
    analyze_stratification("mutagenesis", literals=False)
    analyze_stratification("mutagenesis", literals=True)
