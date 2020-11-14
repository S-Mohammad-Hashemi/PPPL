import sys,os
import numpy as np
import pandas as pd
import torch
from torch import nn
import time
from sklearn.datasets import load_svmlight_files
from os.path import abspath
import scipy.sparse
import pickle
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.model_selection import StratifiedKFold


def _proc_review(doc):
    parts = doc.split(' ')
    label = parts[-1].replace('#label#:', '').strip()
    assert label in ['positive','negative'], 'error parsing label {}'.format(label)
    label = 1 if label == 'positive' else 0
    repeat_word = lambda word, num: ' '.join([word] * int(num))
    text = ' '.join([repeat_word(*x.split(':')) for x in parts[:-1]])
    return text, label

def get_all_docs(dataset_path):
    documents = dict()
    for domain in sorted(os.listdir(dataset_path)):
        documents[domain] = dict()
        for file in ['positive.review', 'negative.review', 'unlabeled.review']:
            documents[domain][file] = []
            for doc in open(os.path.join(dataset_path, domain, file), 'rt'):
                text, label = _proc_review(doc)
                documents[domain][file].append((text,label))
            print('{} documents read for domain {} in file {}'.format(len(documents[domain][file]), domain, file))
    return documents

def _extract_MDS_documents(documents, domain,combine=False):
    pos_docs = [d for d, label in documents[domain]['positive.review']]
    neg_docs = [d for d, label in documents[domain]['negative.review']]
    unlabeled_docs = [d for d, label in documents[domain]['unlabeled.review']]
    unlabeled_y = [label for d,label in documents[domain]['unlabeled.review']]
    unlabeled_docs = np.array(unlabeled_docs)
    unlabeled_y = np.array(unlabeled_y)
    labeled_docs = np.array(pos_docs + neg_docs)
    labels = np.array([1] * len(pos_docs) + [0] * len(neg_docs))
    if combine:
        labeled_docs = np.concatenate((labeled_docs,unlabeled_docs))
        labels = np.concatenate((labels,unlabeled_y))
        return labeled_docs, labels
    return labeled_docs,labels,unlabeled_docs,unlabeled_y

class Vocabulary:
    """
    A bidirectional dictionary words->id and id->words
    """
    def __init__(self, word2idx_dict):
        self._word2idx = word2idx_dict
        self._idx2word = {idx:word for word, idx in word2idx_dict.items()}

    def word2idx(self, word):
        if word in self._word2idx:
            return self._word2idx[word]
        return None

    def idx2word(self, idx):
        if idx in self._idx2word:
            return self._idx2word[idx]
        return None

    def __len__(self):
        return len(self._word2idx)

    def term_set(self):
        return set(self._word2idx.keys())

    def index_list(self):
        return sorted(self._idx2word.keys())

class Domain:
    """
    Defines a domain, composed by a labelled set and a unlabeled set. All sets share a common vocabulary.
    The domain is also characterized by its name and language.
    """

    def __init__(self, X, y, vocabulary, domain, U=None, U_y=None):
        """
        :param X: the document collection
        :param y: the document labels
        :param U: the unlabeled collection
        :param vocabulary: the feature space of X and U
        :param domain: a descriptive name of the domain
        :param language: a descriptive name of the language
        """
        self.X = X
        self.y = y
        self.U = U
        self.U_y = U_y
        self.V=vocabulary if isinstance(vocabulary, Vocabulary) else Vocabulary(vocabulary)
        self.domain = domain


    def show(self):
        print('domain: '+self.domain)
        print('language: ' + self.language)
        print('|V|={}'.format(len(self.V)))
        print('|X|={} (prev={})'.format(self.X.shape[0], self.y.mean()))
        if U is not None:
            print('|U|={}'.format(self.U.shape[0]))

            
def as_domain(labeled_docs, labels, issource, domain, unlabeled_docs=None, unlabeled_y=None, tokken_pattern=r"(?u)\b\w\w+\b", min_df=1):
    """
    Represents raw documents as a Domain; a domain contains the tfidf weighted co-occurrence matrices of the labeled
    and unlabeled documents (with consistent Vocabulary).
    :param labeled_docs: the set of labeled documents
    :param labels: the labels of labeled_docs
    :param unlabeled_docs: the set of unlabeled documents
    :param issource: boolean, if True then the vocabulary is bounded to the labeled documents (the training set), if
    otherwise, then the vocabulary has to be bounded to that of the unlabeled set (which is expecteldy bigger) since
    we should assume the test set is only seen during evaluation. This is not true in a Transductive setting, but we
    force it to follow the same setting so as to allow for a fair evaluation.
    :param domain: the name of the domain (e.g., 'books'
    :param language: the language of the domain (e.g., 'french')
    :param tokken_pattern: the token pattern the sklearn vectorizer will use to split words
    :param min_df: the minimum frequency below which words will be filtered out from the vocabulary
    :return: an instance of Domain
    """
    if issource:
        counter = CountVectorizer(token_pattern=tokken_pattern, min_df=min_df)
        v = counter.fit(labeled_docs).vocabulary_
        tfidf = TfidfVectorizer(sublinear_tf=True, token_pattern=tokken_pattern, vocabulary=v)
    else:
        tfidf = TfidfVectorizer(sublinear_tf=True, token_pattern=tokken_pattern, min_df=min_df)
    if unlabeled_docs is not None:
        X = tfidf.fit_transform(labeled_docs)
        U = tfidf.transform(unlabeled_docs)
        y = np.array(labels)
        V = tfidf.vocabulary_
        U_y = np.array(unlabeled_y)
        domain = Domain(X, y, V, domain, U, U_y)
        return domain
    X = tfidf.fit_transform(labeled_docs)
    y = np.array(labels)
    V = tfidf.vocabulary_
    domain = Domain(X, y, V, domain)
    return domain

def unify_feat_space(source, target):
    """
    Given a source and a target domain, returns two new versions of them in which the feature spaces are common, by
    trivially juxtapossing the two vocabularies
    :param source: the source domain
    :param target: the target domain
    :return: a new version of the source and the target domains where the feature space is common
    """
    word_set = source.V.term_set().union(target.V.term_set())
    word2idx = {w:i for i,w in enumerate(word_set)}
    Vshared = Vocabulary(word2idx)

    def reindexDomain(domain, sharedV):
        V = domain.V
        nD=domain.X.shape[0]
        nF=len(sharedV)
        newX = lil_matrix((nD,nF))
        domainIndexes = np.array(V.index_list())
        sharedIndexes = np.array([sharedV.word2idx(w) for w in [V.idx2word(i) for i in domainIndexes]])
        newX[:,sharedIndexes]=domain.X[:,domainIndexes]
        if domain.U is not None:
            newU = lil_matrix((domain.U.shape[0],nF))
            newU[:,sharedIndexes]=domain.U[:,domainIndexes]
            return Domain(newX.tocsr(),domain.y,sharedV,domain.domain+'_shared',newU.tocsr(),domain.U_y)
#         return Domain(newX.tocsr(),domain.y,None,sharedV,domain.domain+'_shared',domain.language)
        return Domain(newX.tocsr(),domain.y,sharedV,domain.domain+'_shared')

    return reindexDomain(source, Vshared), reindexDomain(target, Vshared)


def preproces_datasets(s_domain, t_domain, documents):
    source_docs, source_labels,_,_ = _extract_MDS_documents(documents, s_domain)
    target_docs, target_labels= _extract_MDS_documents(documents, t_domain,combine=True)
    print('source_docs.shape',source_docs.shape,'source_labels.shape',source_labels.shape,
          'target_docs.shape',target_docs.shape,'target_labels.shape',target_labels.shape)

    source = as_domain(source_docs, source_labels,
                       issource=True, domain=s_domain, min_df=3, unlabeled_docs=None, unlabeled_y=None)
    target = as_domain(target_docs, target_labels,
                       issource=False, domain=t_domain, min_df=3, unlabeled_docs=None, unlabeled_y=None)

    source, target = unify_feat_space(source, target)
    print('Shapes after unifying features:')
    print('source.X.shape',source.X.shape,'target.X.shape',target.X.shape)

    x_source = source.X.astype(np.float32).toarray()
    x_target = target.X.astype(np.float32).toarray()
    x_test = target.X.astype(np.float32)[2000:].toarray()
    y_source = source.y
    y_target = target.y
    y_test = target.y[2000:]

    x_all = np.concatenate((x_source,x_target))
    active_features = x_all!=0
    active_features_count = np.sum(active_features,axis=0)
    most_freq_features = np.argsort(active_features_count)[::-1]
    top_30000_f_inds = most_freq_features[:30000]
    del x_all

    x_source = x_source[:,top_30000_f_inds]
    x_target = x_target[:,top_30000_f_inds]
    x_test = x_test[:,top_30000_f_inds]

    print('Shapes of the inputs that will be fed to the classifier:')
    print('x_source contains positive and negative reviews from the src domain.')
    print('x_target contains positive, negative and unlabeled reviews from the trg domain.')
    print('x_test contains unlabeled reviews from the trg domain. The final accuracy is reported on x_test.')
    print('x_source.shape',x_source.shape,'x_target.shape',x_target.shape,'x_test.shape',x_test.shape)
    
    
    x_source = torch.tensor(x_source,dtype=torch.float32)
    x_target = torch.tensor(x_target,dtype=torch.float32)
    x_test = torch.tensor(x_test,dtype=torch.float32)

    y_source = torch.tensor(y_source,dtype=torch.int64)
    y_target = torch.tensor(y_target,dtype=torch.int64)
    y_test = torch.tensor(y_test,dtype=torch.int64)

    
    return x_source,y_source,x_target,y_target,x_test,y_test




class DataHandler():
    def __init__(self,dataset,labels,weights,batch_size = 64,shuffle=False):
        self.dataset = dataset
        self.current = 0
        self.len = len(dataset)
        self.batch_size = batch_size
        self.labels = labels
        self.do_shuffle = shuffle
        self.inds = np.arange(self.len)
        self.weights = weights
        assert self.len>=batch_size
        assert len(labels)==len(dataset)
        if self.do_shuffle:
            self.shuffle()
    def shuffle(self):
        if self.do_shuffle:
            p = np.random.permutation(len(self.dataset))
            self.inds = p
    def next_batch(self,batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        current_inds = self.inds[self.current:self.current+batch_size]
        batch = self.dataset[current_inds]
        y_batch = self.labels[current_inds]
        if self.weights is not None:
            w_batch = self.weights[current_inds]
        self.current +=batch_size
        if self.current>=self.len:
            new_inds = self.inds[:batch_size-len(batch)]
            batch = torch.cat((batch,self.dataset[new_inds]))
            y_batch = torch.cat((y_batch,self.labels[new_inds]))
            if self.weights is not None:
                w_batch =  torch.cat((w_batch,self.weights[new_inds]))
            self.current=0
            self.shuffle()
        if self.weights is not None:
            return batch,y_batch,w_batch
        return batch,y_batch
    
class SimpleModel(nn.Module):
    def __init__(self,input_size):
        super(SimpleModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size,2048*1),
            nn.ReLU(),
            nn.Linear(2048*1,2048*1),
            nn.ReLU(),
            nn.Linear(2048*1,2048*1),
            nn.ReLU(),
            nn.Linear(2048*1,2)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        logits = self.seq(x)
        probs = self.softmax(logits)
        return logits,probs
    
def adjust_learning_rate_inv(lr, optimizer, iters, alpha=0.001, beta=0.75):
    lr = lr / pow(1.0 + alpha * iters, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 

class Solver():
    def __init__(self,optimizer,net,base_lr):
        self.iters = 0
        self.optimizer = optimizer
        self.net = net
        self.base_lr = base_lr
    def update_lr(self):
        adjust_learning_rate_inv(self.base_lr,self.optimizer,self.iters)