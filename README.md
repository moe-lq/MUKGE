# MUKGE model
Uncertain Knowledge Graph Embedding: An Effective Method Combining Multi-relation and Multi-path

Environment require:
Python 3
tensorflow 1.5.0
scikit-learn

Run experienments:
python ./run.py
python ./run.py --data ppi5k --model rect --batch_size 1024 --dim 128 --epoch 100 --reg_scale 5e-4

# Datasets for benchmark
In every data directory, train.tsv is the training data, test.tsv the testing data and val.tsv the validation data.
|  dataset    |   entities	|  relations  |  asymmetric facts  |  average of confidence  |  standard deviation  |
|   ----      |   ----      |   ----      |         ----       |          ----           |          ----        |
|  CN15k      |   15,000    |     36      |        1,940       |         0.629           |         0.232        |
|  NL27k      |   27,221    |     404     |        2,478       |         0.797           |         0.242        |
|  PPI5k      |   4,999     |     7       |        38,836      |         0.415           |         0.213        |
|  CN15k_ASY  |   15,000    |     36      |        41,190      |         0.606           |         0.321        |
|  NL27k_ASY  |   27,221    |     404     |        98,712      |         0.687           |         0.340        |
|  PPI5k_ASY  |   4,999     |     7       |        226,395     |         0.268           |         0.241        |
