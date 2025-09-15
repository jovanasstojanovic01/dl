from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, add, Dropout # type: ignore

class CaptionModelBuilder:
    def __init__(self, vocab_size, max_len):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def build(self):
        inputs1 = Input(shape = (4096,))
        fe1 = Dropout(0.4)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        inputs2 = Input(shape = (self.max_len,))
        se1 = Embedding(self.vocab_size, 256, mask_zero = True)(inputs2)
        se2 = Dropout(0.4)(se1)
        se3 = LSTM(256)(se1)

        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation = 'relu')(decoder1)
        outputs = Dense(self.vocab_size, activation = 'softmax')(decoder2)

        model = Model(inputs = [inputs1, inputs2], outputs = outputs)
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

        return model
