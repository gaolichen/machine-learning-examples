# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as utils

import utils
import settings
from dataset import MaskedPoetryDataSet, load_poetry_sentences
from model import MaskedPoetryModel

def compute_accuracy(masked_words, predict_words):
    is_correct = [1 if masked_words[i] in pred else 0 for i, pred in enumerate(predict_words)]
    return sum(is_correct) / len(is_correct)

def show_prediction(masked_sentences, masked_words, predict_words, predict_prob):
    correct_predicts = []
    wrong_predicts = []

    for sen, expected, retured, prob in zip(masked_sentences, masked_words, predict_words, predict_prob):
        if expected in retured:
            correct_predicts.append((sen, expected, list(zip(retured, prob))))
        else:
            wrong_predicts.append((sen, expected, list(zip(retured, prob))))

    print('correct predicts:')
    for sen in correct_predicts:
        text = utils.ljust(sen[0].replace('[MASK]', '__'), 40)
        prediction = '[({0},{1:.3f}), ({2}, {3:.3f})]'.format(sen[2][0][0], sen[2][0][1], sen[2][1][0], sen[2][1][1])
        print('{0}{1:<4}\tprediction: {2:<20}'.format(text, sen[1], prediction))

    print()
    print('incorrect predicts:')
    for sen in wrong_predicts:
        text = utils.ljust(sen[0].replace('[MASK]', '__'), 40)
        prediction = '[({0},{1:.3f}), ({2}, {3:.3f})]'.format(sen[2][0][0], sen[2][0][1], sen[2][1][0], sen[2][1][1])
        print('{0}{1:<4}\tprediction: {2:<20}'.format(text, sen[1], prediction))

def predict_and_show(model, masked_sentence, masked_words, topk = 2):
    predict_words, predict_prob = model.predict(masked_sentence, topk = topk)
    accuracy = compute_accuracy(masked_words, predict_words)
    print('accuracy =', accuracy)
    show_prediction(masked_sentence, masked_words, predict_words, predict_prob)


tang_poetry_size = 1200
song_poetry_size = 6000

def process():
    print('Loading poetry...')
    tr_poetry_sentences, val_poetry_sentences = load_poetry_sentences(tang_poetry_size, song_poetry_size)


    train_ds = MaskedPoetryDataSet(tr_poetry_sentences,
                                   input_dim = settings.input_dim,
                                   batch_size = settings.batch_size)

    val_ds = MaskedPoetryDataSet(val_poetry_sentences,
                                 input_dim = settings.input_dim,
                                 batch_size = settings.batch_size)

    print('Building model...')
    bert_mlm = utils.load_bert_model()

    model = MaskedPoetryModel(bert_mlm, settings.input_dim)

    model.prepare(len(train_ds))

    sample_masked_sentence, sample_masked_words = utils.to_masked_sentence(val_poetry_sentences[:100])
    predict_and_show(model, sample_masked_sentence, sample_masked_words)

    print('training...')
    model.fit(train_ds, epochs = settings.epochs, validation_data = val_ds)

    predict_and_show(model, sample_masked_sentence, sample_masked_words)


if __name__ == '__main__':
    process()
