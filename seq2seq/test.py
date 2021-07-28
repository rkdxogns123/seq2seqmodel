
from seq2seq.model import *

model1 = seq2seq()

model1.load_weights('C:\\learn\\vocab\\out\\', 'seq2seq_kor', 'weights.h5')

model1.save('testing.h5')