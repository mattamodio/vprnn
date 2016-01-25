import os
import glob
from collections import Counter
import numpy as np

sentences = []
PATH = "/u/c/j/cjsmith/vprnn/code-corpora/c/git/"
for file in glob.glob(os.path.join(PATH, "*.c")):
    print "Reading file " + file
    sentence = ["START"] + list(open(file, 'r').read()) + ["END"]
    sentences.append(sentence)

print "All files read"
print "Constructing data..."
data = []
for sentence in sentences:
    data += sentence
chars = Counter(data)
sorted_chars = sorted(chars.keys(), key=lambda c: chars[c])
INPUT_LENGTH = len(chars)
char_map = {c : i for i, c in enumerate(sorted(chars.keys(), key=lambda c: chars[c]))}
min_sentence_length = min([len(s) for s in sentences])

def char_to_vector(char):
    output = np.zeros(INPUT_LENGTH)
    output[char_map[char]] = 1
    return output

def vector_to_char(v):
    return sorted_chars[v.index(max(v))]

x_train = np.empty( (len(sentences), min_sentence_length - 1, INPUT_LENGTH) )
y_train = np.empty( (len(sentences), min_sentence_length - 1) )

for i, sentence in enumerate(sentences):
    for j in range(min_sentence_length - 1):
        x_train[i][j] = char_to_vector(sentence[j])
        y_train[i][j] = char_map[sentence[j+1]]
