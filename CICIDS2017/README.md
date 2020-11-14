# PPPL on the CICIDS2017 dataset

CICIDS_PPPL.ipynb contains the code to train a model on the CICIDS2017 dataset with the help of the PPPL technique.
This code is developed using TensorFlow.

For more details on how this dataset is created and preprocessed, please look at [this paper](https://arxiv.org/abs/2008.03677).

Since this dataset is imbalanced, at the initial phase of the training, when we train the model on the source samples, we first downsampled the benign inputs
to become the same size as malicious inputs and then trained the model on them. 
During the first 30 iterations of PPPL, we balanced benign and malicious inputs by downsampling benign inputs, as well.
In addition during the first 30 iterations of PPPL, instead of excluding samples of the class which is predicted more than its expected class proportion
since there are only two classes, we changed their pseudo-labels to the opposite class and keep them in the training set. 
We do these two techniques because of the imbalanced domains that we have in this dataset to force the model to predict malicious inputs more often 
and increase the chance of correct predictions of malicious inputs.

To do domain adaptation between any pair of the domains in this dataset, s_domain and t_domain variables in the first cell of the notebook should be set appropriately.

Dependencies:
```
python 3.7
tensorflow 2.1.0
sklearn 0.22.2
pandas 1.0.3
```
