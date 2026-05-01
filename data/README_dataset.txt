Word2Vec Final Project Dataset
================================

Dataset type:
Instructor-prepared educational English corpus for Skip-gram / Word2Vec from scratch.

Why this dataset:
- standardized for the whole class
- lowercase and simple tokenization friendly
- large enough for meaningful co-occurrence learning
- still small enough for NumPy-based experiments

Basic statistics:
- sentences: 7478
- tokens: 57216
- vocabulary size: 164

Recommended usage:
- Part A (baseline full softmax): train on a reduced subset of this corpus
- Part B (negative sampling): train on the full corpus

Permitted preprocessing:
- lowercase
- simple tokenization
- remove empty lines
- optional min_count filtering

Not recommended:
- aggressive linguistic preprocessing
- replacing the dataset
- web crawling or collecting a new dataset for this assignment
