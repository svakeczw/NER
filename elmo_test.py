# coding: utf-8

import os
import pickle
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from utils.feeder.LSTMCNNCRFeeder import LSTMCNNCRFeeder
from utils.parser import parse_conll2003
from utils.conlleval import evaluate
from utils.checkmate import best_checkpoint

from model.Elmo import ElmoModel

from bilm import Batcher, BidirectionalLanguageModel


def conll2003():
    if not os.path.isfile('dev/conll.pkl'):
        parse_conll2003()
    with open('dev/conll.pkl', 'rb') as fp:
        train_set, val_set, test_set, dicts = pickle.load(fp)

    return train_set, val_set, test_set, dicts


train_set, val_set, test_set, dicts = conll2003()

w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']
idx2w = {w2idx[k]: k for k in w2idx}
idx2la = {la2idx[k]: k for k in la2idx}

train_x, train_chars, train_la = train_set
val_x, val_chars, val_la = val_set
test_x, test_chars, test_la = test_set

print('Load elmo...')
elmo_batcher = Batcher('dev/vocab.txt', 50)
elmo_bilm = BidirectionalLanguageModel('resources/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
                                       'resources/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

print('Load model...')

num_classes = len(la2idx.keys())
max_seq_length = max(
    max(map(len, train_x)),
    max(map(len, test_x)),
)
max_word_length = max(
    max([len(ssc) for sc in train_chars for ssc in sc]),
    max([len(ssc) for sc in test_chars for ssc in sc])
)

model = ElmoModel(
    True,
    50,  # Word embedding size
    16,  # Character embedding size
    200,  # LSTM state size
    128,  # Filter num
    3,  # Filter size
    num_classes,
    max_seq_length,
    max_word_length,
    0.015,
    0.5,
    elmo_bilm,
    1,  # elmo_mode
    elmo_batcher)

print('Start training...')
print('Train size = %d' % len(train_x))
print('Val size = %d' % len(val_x))
print('Test size = %d' % len(test_x))
print('Num classes = %d' % num_classes)

start_epoch = 1
max_epoch = 100

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

best_checkpoint = best_checkpoint('checkpoints/best/', True)
sess.run(tf.tables_initializer())
saver.restore(sess, best_checkpoint)

train_feeder = LSTMCNNCRFeeder(train_x, train_chars, train_la, max_seq_length, max_word_length, 16)
val_feeder = LSTMCNNCRFeeder(val_x, val_chars, val_la, max_seq_length, max_word_length, 16)
test_feeder = LSTMCNNCRFeeder(test_x, test_chars, test_la, max_seq_length, max_word_length, 16)

'''
preds = []
for step in tqdm(range(val_feeder.step_per_epoch)):
    tokens, chars, labels = val_feeder.feed()
    pred = model.test(sess, tokens, chars)
    preds.extend(pred)
true_seqs = [idx2la[la] for sl in val_la for la in sl]
pred_seqs = [idx2la[la] for sl in preds for la in sl]
ll = min(len(true_seqs), len(pred_seqs))
_, _, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

val_feeder.next_epoch(False)

print("\nval_f1: %f" % f1)

preds = []
for step in tqdm(range(test_feeder.step_per_epoch)):
    tokens, chars, labels = test_feeder.feed()
    pred = model.test(sess, tokens, chars)
    preds.extend(pred)
true_seqs = [idx2la[la] for sl in test_la for la in sl]
pred_seqs = [idx2la[la] for sl in preds for la in sl]
ll = min(len(true_seqs), len(pred_seqs))
_, _, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

test_feeder.next_epoch(False)

print("\ntest_f1: %f" % f1)
'''


def dump_topK(prefix, feeder, topK):
    """
    TOKEN LABEL TOP1 TOP2 ... TOPN B_PER I_PER
    """
    with open('dev/predict.%s' % prefix, 'w') as fp:

        fp.write('\t'.join(['TOKEN', 'LABEL'] +
                           ['TOP_%d' % (i + 1) for i in range(topK)] +
                           list(la2idx.keys())) + '\n\n')

        for _ in tqdm(range(feeder.step_per_epoch)):
            tokens, chars, labels = feeder.feed()

            out_seqs, out_path_scores, out_position_scores = model.decode(sess, tokens, chars, topK)
            for i, (preds, path_scores, position_scores) in enumerate(
                    zip(out_seqs, out_path_scores, out_position_scores)):
                length = len(preds[0])

                st = tokens[i, :length].tolist()
                sl = [idx2la[la] for la in labels[i, :length].tolist()]

                # Position score
                norm_position_scores = np.zeros(shape=(num_classes, length))
                for pred, position_score in zip(preds, position_scores):
                    for t in range(length):
                        norm_position_scores[pred[t], t] += position_score[t]
                norm_position_scores = norm_position_scores.tolist()

                e = np.array(norm_position_scores)
                e = e / e.sum(axis=0, keepdims=True)
                norm_position_scores = [['{:.4f}'.format(e2) for e2 in e1] for e1 in e]

                # Top N
                preds = [[idx2la[la] for la in pred] for pred in preds]

                for all in zip(*[st, sl, *preds, *norm_position_scores]):
                    fp.write('\t'.join(all) + '\n')

                # Path scores
                score_all = sum(path_scores)

                norm_path_scores = [''] * 2 + \
                                   ['{:.4f}'.format(score / score_all) for score in path_scores] + \
                                   [''] * topK

                path_scores = [''] * 2 + \
                              ['{:.4f}'.format(score) for score in path_scores] + \
                              [''] * topK

                fp.write('\t'.join(path_scores) + '\n')
                fp.write('\t'.join(norm_path_scores) + '\n')

                fp.write('\n')


def restore_zeros(prefix):
    # Restore zeros
    with open('data/%s.txt' % prefix) as fin1:
        with open('dev/predict.%s' % prefix) as fin2:
            with open('eval/predict.%s' % prefix, 'w') as fout:
                fout.write(fin2.readline())
                fout.write(fin2.readline())

                for row1 in fin1:
                    if row1 == '\n':
                        fout.write(fin2.readline())
                        fout.write(fin2.readline())
                        fout.write(fin2.readline())
                    else:
                        row2 = fin2.readline()
                        data1 = row1.split(' ')
                        data2 = row2.split('\t')
                        data2[0] = data1[0]
                        fout.write('\t'.join(data2))


# dump_topK('train', train_feeder, 10)
# dump_topK('valid', val_feeder, 10)
dump_topK('test', test_feeder, 10)

# restore_zeros('train')
# restore_zeros('valid')
restore_zeros('test')
