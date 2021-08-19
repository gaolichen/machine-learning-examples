# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

from dataclasses import dataclass
import tensorflow as tf

from datasets import ChinesePoetry, is_ch
from masked_language_model.dataset import MaskedPoetryDataSet, to_masked_sentence
from masked_language_model.model import MaskedPoetryModel

#import os
#import sys
#current = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(os.path.join(current, '../../../simplebert/src'))


from simplebert import tokenizer_from_pretrained, model_from_pretrained
    

def text_width(text):
  w = len(text)
  for ch in text:
    if is_ch(ch):
      w += 1
  return w

def rjust(text, width):
  len = text_width(text)
  return (' ' * (width - len)) + text

def ljust(text, width):
  len = text_width(text)
  return text + (' ' * (width - len))

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
        text = ljust(sen[0].replace('[MASK]', '__'), 40)
        prediction = '[({0},{1:.3f}), ({2}, {3:.3f})]'.format(sen[2][0][0], sen[2][0][1], sen[2][1][0], sen[2][1][1])
        print('{0}{1:<4}\tprediction: {2:<20}'.format(text, sen[1], prediction))

    print()
    print('incorrect predicts:')
    for sen in wrong_predicts:
        text = ljust(sen[0].replace('[MASK]', '__'), 40)
        prediction = '[({0},{1:.3f}), ({2}, {3:.3f})]'.format(sen[2][0][0], sen[2][0][1], sen[2][1][0], sen[2][1][1])
        print('{0}{1:<4}\tprediction: {2:<20}'.format(text, sen[1], prediction))

def predict_and_show(model, masked_sentence, masked_words, topk = 2):
    predict_words, predict_prob = model.predict(masked_sentence, topk = topk)
    accuracy = compute_accuracy(masked_words, predict_words)
    print('accuracy =', accuracy)
    show_prediction(masked_sentence, masked_words, predict_words, predict_prob)

@dataclass
class Settings(object):
    epochs: int = 5
    batch_size: int = 32
    input_dim: int = 64
    learning_rate: float = 5e-5
    model_name: str = 'bert-base-chinese'
    val_split: float = 0.1
    save_dir: str = '.'
    download_size: float = 1.0
    

def process(settings):
    print('Loading poetry...')
    cp = ChinesePoetry()
    tr_poetry_sentences, val_poetry_sentences = cp.load_dataset('song',
                                                                whole_poetry = False,
                                                                maxlen = settings.input_dim,
                                                                val_split = settings.val_split,
                                                                download_size = settings.download_size)
    
    tr_poetry_sentences2, val_poetry_sentences2 = cp.load_dataset('tang',
                                                                whole_poetry = False,
                                                                maxlen = settings.input_dim,
                                                                val_split = settings.val_split,
                                                                download_size = settings.download_size)

    tr_poetry_sentences.extend(tr_poetry_sentences2)
    val_poetry_sentences.extend(val_poetry_sentences2)
    tokenizer = tokenizer_from_pretrained(settings.model_name)


    train_ds = MaskedPoetryDataSet(tr_poetry_sentences,
                                   tokenizer = tokenizer,
                                   settings = settings)
    val_ds = MaskedPoetryDataSet(val_poetry_sentences,
                                   tokenizer = tokenizer,
                                   settings = settings)

    print('Building model...')
    bert_mlm = model_from_pretrained(settings.model_name, causal_attention = False, model_head = 'lm')
    model = MaskedPoetryModel(bert_mlm, tokenizer = tokenizer, settings = settings)

    model.prepare(len(train_ds))

    sample_masked_sentence, sample_masked_words = to_masked_sentence(val_poetry_sentences[:100])
    predict_and_show(model, sample_masked_sentence, sample_masked_words)

    print('training...')
    model.fit(train_ds, epochs = settings.epochs, validation_data = val_ds)

    predict_and_show(model, sample_masked_sentence, sample_masked_words)


if __name__ == '__main__':
    process(Settings())
