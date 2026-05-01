From Skip-gram Word2Vec from Scratch to Practical Word
Embedding Evaluation Project
======================

Required file structure
-----------------------

- baseline_skipgram.py
- negative_sampling.py
- evaluate.py
- train.py
- loss_curve_baseline.png
- loss_curve_neg_sampling.png
- README.txt
- data/corpus.txt
- results/baseline_output.txt
- results/negative_sampling_output.txt
- results/part_c_output.txt
- results/part_d_output.txt
- results/evaluation_output.txt

Python version
--------------

- Recommended: Python 3.11 or newer

Dependencies
------------

- numpy
- matplotlib
- scipy
- gensim

Install
-------

python -m pip install numpy matplotlib scipy gensim

How to run
----------

Run all required training experiments:

python train.py

Run intrinsic evaluation and Gensim comparison:

python evaluate.py

Suggested execution order
-------------------------

1. Run `python train.py`
2. Run `python evaluate.py`
3. Use the files in `results/` together with the loss curves for the report

What train.py does
------------------

1. Runs Part A baseline full-softmax Skip-gram on a reduced subset of the provided corpus.
2. Saves the baseline loss curve to loss_curve_baseline.png.
3. Runs Part B Negative Sampling Skip-gram on the full provided corpus.
4. Saves the Negative Sampling loss curve to loss_curve_neg_sampling.png.
5. Writes text outputs to results/baseline_output.txt and results/negative_sampling_output.txt.

What evaluate.py does
---------------------

1. Re-runs the baseline model on the reduced subset.
2. Re-runs the Negative Sampling model on the full corpus.
3. Computes cosine similarities for selected evaluation pairs.
4. Computes top-3 nearest neighbors for selected query words.
5. Trains a matched Gensim Word2Vec model on the same full corpus.
6. Writes Part C and Part D outputs to:
   - results/part_c_output.txt
   - results/part_d_output.txt
   - results/evaluation_output.txt

Output files to include in the submission
-----------------------------------------

- loss_curve_baseline.png
- loss_curve_neg_sampling.png
- results/baseline_output.txt
- results/negative_sampling_output.txt
- results/part_c_output.txt
- results/part_d_output.txt
- results/evaluation_output.txt

Dataset usage
-------------

- Part A baseline uses a reduced subset of 2000 sentences for correctness verification and controlled training.
- Part B Negative Sampling uses the full standardized corpus in data/corpus.txt.

Default hyperparameters
-----------------------

Baseline:
- embedding dimension = 50
- context window = 2
- epochs = 5
- initial learning rate = 0.025
- learning-rate decay = 0.005

Negative Sampling:
- embedding dimension = 50
- context window = 2
- epochs = 5
- initial learning rate = 0.025
- learning-rate decay = 0.0
- negative samples K = 5

Notes
-----

- Full softmax is kept for the baseline because it is pedagogically useful and feasible on a reduced subset.
- Negative Sampling is the practical extension for full-corpus training because each update uses only one positive context word and a small set of sampled negative words.
