from preprocess import *

PATH = 'C:\\learn\\ChatBotData.csv_short.csv'
VOCAB_PATH = 'C:\\learn\\vocab\\vocabulary1.txt'

inputs, outputs = load_data(PATH)

char2idx, idx2char, vocab_size = load_vocabulary(PATH, VOCAB_PATH, tokenize_as_morph=False)

index_inputs, input_seq_len = enc_processing(inputs, char2idx, tokenize_as_morph=False)
index_outputs, output_seq_len = dec_output_processing(outputs, char2idx, tokenize_as_morph=False)
index_targets = dec_target_processing(outputs, char2idx, tokenize_as_morph=False)

data_configs = {}
data_configs['char2idx'] = char2idx
data_configs['idx2char'] = idx2char
data_configs['vocab_size'] = vocab_size
data_configs['pad_symbol'] = PAD
data_configs['std_symbol'] = STD
data_configs['end_symbol'] = END
data_configs['unk_symbol'] = UNK

DATA_IN_PATH = 'C:\\learn\\vocab\\in\\'
TRAIN_INPUTS = 'train_inputs1.npy'
TRAIN_OUTPUTS = 'train_outputs1.npy'
TRAIN_TARGETS = 'train_targets1.npy'
DATA_CONFIGS = 'data_configs1.json'

np.save(open(DATA_IN_PATH + TRAIN_INPUTS, 'wb'), index_inputs)
np.save(open(DATA_IN_PATH + TRAIN_OUTPUTS , 'wb'), index_outputs)
np.save(open(DATA_IN_PATH + TRAIN_TARGETS , 'wb'), index_targets)

json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'))