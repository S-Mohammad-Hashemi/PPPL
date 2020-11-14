# General Domain Adaptation Through Proportional Progressive Pseudo Labeling

This repository contains the code to train a model with the Proportional Progressive Pseudo Labeling (PPPL) technique as described in our paper.

Before running the code for CICIDS2017, the dataset should be first extracted.
```
./extract_packet_data.sh
```

For more details about the code for each dataset please look at its own README file.

## Paper

**Abstract:**

Domain adaptation helps transfer the knowledge gained from a labeled source domain to an unlabeled target domain. During the past few years, different domain adaptation techniques have been published. One common flaw of these approaches is that while they might work well on one input type, such as images, their performance drops when applied to others, such as text or time-series. In this paper, we introduce Proportional Progressive Pseudo Labeling (PPPL), a simple, yet effective technique that can be implemented in a few lines of code to build a more general domain adaptation technique that can be applied on several different input types. At the beginning of the training phase, PPPL progressively reduces target domain classification error, by training the model directly with pseudo-labeled target domain samples, while excluding samples with more likely wrong pseudo-labels from the training set and also postponing training on such samples. Experiments on 6 different datasets that include tasks such as anomaly detection, text sentiment analysis and image classification demonstrate that PPPL can beat other baselines and generalize better.

## Citation
```
@inproceedings{pppl,
  title={General Domain Adaptation Through Proportional Progressive Pseudo Labeling},
  author={Mohammad J. Hashemi and Eric Keller},
  booktitle={2020 IEEE International Conference on Big Data (Big Data)},
  year={2020}
}
```

Questions? Please, email mohammad.hashemi@colorado.edu.
