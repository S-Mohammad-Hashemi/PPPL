# PPPL on the MDS dataset
MDS_PPPL.ipynb contains the code to train a model on the MDS dataset with the help of the PPPL technique. 
This code is developed using PyTorch. 

The MDS dataset contains 27,677 product reviews from amazon.com about four product domains: books (B), dvds (D), electronics (E) and kitchen appliances (K). 
The goal is to classify the reviews into positive and negative classes. 
For each domain, 2,000 reviews are named labeled and around 4,000 (4,465 for books, 3,586 for dvds, 5,681 for electronics and 5,945 for kitchen appliances) 
are named unlabeled. 

To do domain adaptation between any pair of the above domains s_domain and t_domain variables in the first cell of the notebook should be set appropriately.

During the domain adaptation phase, we used 2,000 labeled reviews from the source domain
and both of the labeled and unlabeled reviews, from the target domain (without their labels). 
The final classification accuracy is reported on the unlabeled reviews of the target domain. 

For preprocessing this dataset, 
we used some of the code from [here](https://github.com/AlexMoreo/pydci).
Essentially, we used the Bag of Words (BoW) method, then selected 30,000 features that were most frequent among the reviews in each task and
discarded the others and fed them to the model. 

Dependencies:
```
python 3.7
pytorch 1.5.0
torchvision 0.6.0
sklearn 0.22.2
scipy 1.4.1
```
