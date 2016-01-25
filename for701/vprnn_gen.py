from data import *
from archs import MetaVPRNN
import numpy as np

model = MetaVPRNN(n_in=INPUT_LENGTH, n_hidden=150, n_out=INPUT_LENGTH,
        learning_rate=0.01, learning_rate_decay=0.999, n_epochs=200,
        activation='tanh', output_type='softmax', use_symbolic_softmax=False)
model.load("models/vprnn.model")

def gen(model, seed = None):
    if seed:
        observed = [char_to_vector("START")] + [char_to_vector(c) for c in seed]
    else:
        observed = [char_to_vector("START")]
    output = ["START"]
    for i in range(200):
        seen = np.where(np.random.multinomial(1, model.predict_proba(observed)[-1]))[0][0]
        observed.append(char_to_vector(sorted_chars[seen]))
        output.append(sorted_chars[seen])
    return "".join(output)

print gen(model)
