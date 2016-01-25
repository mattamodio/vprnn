from data import *
from archs import MetaVPRNN
import logging

model = MetaVPRNN(n_in=INPUT_LENGTH, n_hidden=150, n_out=INPUT_LENGTH,
        learning_rate=0.001, learning_rate_decay=0.999, n_epochs=400,
        activation='tanh', output_type='softmax', use_symbolic_softmax=False)

model.vprnn.PUSH = char_to_vector("{")
model.vprnn.POP = char_to_vector("}")

print "Model construction/training..."
logging.basicConfig(level=logging.INFO, filename="./logs/vprnn.log")
model.fit(x_train, y_train, validation_frequency=344)
model.save(fname="models/vprnn2.model")
