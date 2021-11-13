"""
Script that trains TF multitask models on MUV dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from muv_datasets import load_muv
from graph_model import GraphConvModel
import metrics
import models
import optimizers

def get_featurizer(model_name):
  if model_name == 'graphconv':
    return 'graphconv'
  else:
    return 'ECFP'

def create_model(model_name, n_tasks, batch_size = 64):
  rate = optimizers.ExponentialDecay(0.001, 0.8, 1000)

  if model_name == 'graphconv':
    return GraphConvModel(
        n_tasks,
        learning_rate = rate,
        batch_size=batch_size,
        mode='classification')
    
  elif model_name == 'tf':
    return models.MultitaskClassifier(
        n_tasks,
        n_features=1024,
        dropouts=[.25],
        learning_rate=rate,
        weight_init_stddevs=[.1], 
        batch_size=64,
        verbosity="high")
  else:
    return None  

def run_muv(model_name, epochs = 10):
    print('loading data')
    # Load MUV data
    np.random.seed(123)
    muv_tasks, muv_datasets, transformers = load_muv(
        splitter='stratified',
        featurizer=get_featurizer(model_name))
    train_dataset, valid_dataset, test_dataset = muv_datasets

    print('building model')
    model = create_model(model_name, len(muv_tasks))

    print('fitting...')
    # Fit trained model
    loss = model.fit(train_dataset, nb_epoch=epochs)
    print(f'loss={loss}')

    metric = metrics.Metric(metrics.roc_auc_score, np.mean, mode="classification")
    # Evaluate train/test scores
    print('evaluating...')
    train_scores = model.evaluate(train_dataset, [metric], transformers)
    valid_scores = model.evaluate(valid_dataset, [metric], transformers)
    return train_scores, valid_scores

def plot_muv_scores(scores):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    model_names = list(scores.keys())
    roc_auc = [scores[name] for name in model_names]

    y_pos = np.arange(len(model_names))

    ax.barh(y_pos, roc_auc, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('roc-auc')
    ax.set_xticks([i*0.1 for i in range(11)])
    ax.set_title('Performance on muv')

    for i, v in enumerate(roc_auc):
        ax.text(v + 0.01, i + .1, str(round(v, 3)), color='blue', fontweight='bold')

    plt.show()


train_scores_tf, valid_scores_tf = run_muv('tf', epochs = 10)
print('tf Train scores', train_scores_tf)
print('tf Validation scores', valid_scores_tf)

train_scores_gc, valid_scores_gc = run_muv('graphconv', epochs = 10)
print('graphconv Train scores', train_scores_gc)
print('graphconv Validation scores', valid_scores_gc)

res = {'tf': valid_scores_tf['mean-roc_auc_score'], 'graphconv': valid_scores_gc['mean-roc_auc_score']}
plot_muv_scores(res)
