from data import *
from archs import MetaRNN
import logging

model = MetaRNN(n_in=INPUT_LENGTH, n_hidden=150, n_out=INPUT_LENGTH,
        learning_rate=0.001, learning_rate_decay=0.999, n_epochs=300,
        activation='tanh', output_type='softmax', use_symbolic_softmax=False)

print "Model construction/training..."
logging.basicConfig(level=logging.INFO, filename="logs/rnn.logs")
model.fit(x_train, y_train, validation_frequency=344)
model.save(fname="models/rnn.model")
