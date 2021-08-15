# -*- coding: utf-8 -*-
# @File    : model.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class MaskedPoetryModel(object):
    def __init__(self, bert_mlm, tokenizer, settings):
        self.bert_mlm = bert_mlm
        self.tokenizer = tokenizer
        self.settings = settings
        
        input_ids = keras.Input(shape = (self.settings.input_dim, ), dtype='int32', name = 'input_ids')
        attention_mask = tf.where(input_ids == self.tokenizer.pad_token_id, 0.0, 1.0)
        output = self.bert_mlm({'input_ids':input_ids, 'attention_mask':attention_mask})
        mask_positions = tf.gather(tf.where(input_ids == self.tokenizer.mask_token_id), indices = 1, axis = -1)
        output = tf.gather(output['logits'], mask_positions, axis = 1, batch_dims = 1)
        self.model = keras.models.Model(input_ids, output)

    def prepare(self, batches):
#        optimizer, _ = create_optimizer(
#                          init_lr = 3e-5,
#                          weight_decay_rate = 0.01,
#                          num_train_steps = int(batches * settings.epochs),
#                          num_warmup_steps = int(0.1 * batches))

        self.model.compile(optimizer = keras.optimizers.Adam(learning_rate = self.settings.learning_rate),
                    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        return self.model

    def fit(self, train_ds, *args, **kwargs):
        hist = self.model.fit(train_ds, *args, **kwargs)
        return hist

    def predict(self, masked_sentence, topk = 1):
        input_ids = self.tokenizer.encode(masked_sentence, maxlen = self.settings.input_dim)
        masked_pos = np.where(input_ids == self.tokenizer.mask_token_id)[1]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).astype('int32')
        output = self.bert_mlm([input_ids, attention_mask])
        logits = tf.gather(output['logits'], masked_pos, axis = 1, batch_dims = 1)
        prob = tf.nn.softmax(logits, axis = -1)
        indices = tf.argsort(prob, axis = -1, direction = 'DESCENDING')
        indices = indices[:, :topk]
        flat_indices = tf.reshape(indices, shape = (indices.shape[0] * topk, )).numpy()

        tokens = np.array(self.tokenizer.ids_to_tokens(flat_indices))
        tokens = np.reshape(tokens, indices.shape)
        return tokens.tolist(), tf.gather(prob, indices, axis = 1, batch_dims = 1).numpy().tolist()
