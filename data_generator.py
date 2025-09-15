import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

class DataGenerator:
    def __init__(self, mapping, features, tokenizer, max_len, vocab_size, batch_size):
        self.mapping = mapping
        self.features = features
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def data_generator(self, data_keys):
        n = 0
        X1, X2, Y = list(), list(), list()
        while True:
            for key in data_keys:
                n = n + 1
                for caption in self.mapping[key]:
                    sequence = self.tokenizer.texts_to_sequences([caption])[0]
                    for i in range(1, len(sequence)):
                        input_seq, out_seq = sequence[:i], sequence[i]
                        input_seq = pad_sequences([input_seq], maxlen = self.max_len, padding = 'post')[0]
                        out_seq = to_categorical([out_seq], num_classes = self.vocab_size)[0]
                        X1.append(self.features[key][0])
                        X2.append(input_seq)
                        Y.append(out_seq)

                if n == self.batch_size:
                    n = 0
                    X1, X2, Y = np.array(X1), np.array(X2), np.array(Y)
                    yield (X1, X2), Y

                    X1, X2, Y = list(), list(), list()

    def generator(self, data_keys):
        return self.data_generator(data_keys)
