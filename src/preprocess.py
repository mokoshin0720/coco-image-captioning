import nltk
import pickle
import os
import numpy as np
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
import glob
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.utils.data as data

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def make_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()

    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}{}] Tokenized captions.".format(i+1, len(ids)))
        
        words = [word for word, cnt in counter.items() if cnt >= threshold]

        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        for i, word in enumerate(word):
            vocab.add_word(word)
        
        return vocab

if __name__ == '__main__':
    vocab = make_vocab(
        json='/data/annotations/captions_train2014.json',
        threshold=4,
    )
    vocab_path = '/data/vocab.pkl'
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved vocabulary wrapper to '{}'".format(vocab_path))