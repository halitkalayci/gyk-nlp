import torch
import torch.nn as nn
import torch.nn.functional as F

sentences = [
    "ali okula gitti",
    "veli sinemaya gitti",
    "ayşe markete gitti",
    "fatma işe gitti"
]

# 1. Tokenization (manual)

vocab = { "<PAD>": 0 } # <PAD> => Padding

index = 1

for sentence in sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = index
            index += 1

#

# encode()
def encode(sentence):
    tokens = sentence.split()
    # cümlenin son kelimesi hariç tamamı - son kelimesi
    return torch.tensor([vocab[token] for token in tokens[:-1]], dtype=torch.long), torch.tensor(vocab[tokens[-1]], dtype=torch.long)
#

data = [encode(sentence) for sentence in sentences]
