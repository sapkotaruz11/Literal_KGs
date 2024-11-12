# Literal_KGs
A repository of collection of knowledge graphs with literal values for predicting literals (numeric, texts) and knowledge graph completion.


## Current datasets

- DB15k
- FB15k
- Literally Wikidata
    - 1k
    - 19k
    - 48k

- YAGO15k

## Dataset Structures

- EntityTriples.txt ---> All non-literal triples for learning embeddings ( train, test, val ) combined
- NumericTriples --> Unfiltered Numeric Literal data
- NumericTriples_filtered --> processed Numeric Literal data ( ensure entity in EntityTriple, fix relation names, extract only the numeric value for tails etc)
- train, test, val --> train, test, val split of filtered numeric literals (suffeled, 80/10/10 split)


## Dataset Statistics 

| Split           | DB15k   | FB15k-237  | YAGO    |
|-----------------|---------|------------|---------|
| Entity Triples  | 99,028  | 310,116    | 122,886 |
| Train           | 38,484  | 18,428     | 18,826  |
| Test            | 4,811   | 2,304      | 2,354   |
| Validation      | 4,810   | 2,305      | 2,355   |
----------------------------------------------------
train/test/validation refer to the split of Numeric Values for literal prediction task
FB15k-237 has total of 48105 numeric triples with 116 different relations, only the top 10 relations are considered for literal prediction task ( as done in [Learning Numerical Attributes in Knowledge Bases](https://www.akbc.ws/2019/papers/BJlh0x9ppQ) 