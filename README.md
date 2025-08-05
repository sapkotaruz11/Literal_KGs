# Literal_KGs
A repository of collection of knowledge graphs with literal values for predicting literals (numeric, texts) and knowledge graph completion.


## Current datasets

- FB15k-237-lit
- Literally Wikidata
    - 1k
    - 19k
    - 48k

- YAGO10 plus

## Dataset Structures

- EntityTriples.txt ---> All non-literal triples for learning embeddings ( train, test, val ) combined
- NumericTriples --> Unfiltered Numeric Literal data
- NumericTriples_filtered --> processed Numeric Literal data ( ensure entity in EntityTriple, fix relation names, extract only the numeric value for tails etc)
- train, test, val --> train, test, val split of filtered numeric literals (suffeled, 80/10/10 split)


## Dataset Statistics 

| Split           | DB15k   | FB15k-237  | YAGO15k   |
|-----------------|---------|------------|---------------|
| Entity Triples  | 99,028  | 310,116    | 138,056     |
| Train           | 9,464  | 18, 615     |   18,825      |
| Test            |  1,183  | 2,329      | 2,354        |
| Validation      | 1,184   | 2,306     | 2,354         |
----------------------------------------------------------
train/test/validation refer to the split of Numeric Values for literal prediction task
