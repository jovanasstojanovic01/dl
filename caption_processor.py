from tqdm import tqdm
import pickle
import re

from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore

class CaptionProcessor:
    def __init__(self, captions_file, mapping_file, tokenizer_file):
        self.captions_file = captions_file
        self.mapping_file = mapping_file
        self.tokenizer_file = tokenizer_file
        self.mapping = {}

    def load_captions(self):
        with open(self.captions_file, "r") as f:
            next(f)
            captions_doc = f.read()

        return captions_doc

    def build_mapping(self):
        captions_doc = self.load_captions()
        for line in tqdm(captions_doc.split("\n")):
            if len(line) < 2:
                continue

            tokens = line.split(",")
            img, caption = tokens[0], tokens[1:]
            img_id = img.split(".")[0]
            caption = " ".join(caption)
            if img_id not in self.mapping:
                self.mapping[img_id] = []

            self.mapping[img_id].append(caption)

        with open(self.mapping_file, "wb") as f:
            pickle.dump(self.mapping, f)

        with open(self.mapping_file, "rb") as f:
            pickle.load(f)
    
    def clean(self):
        for _, captions in self.mapping.items():
            for i in range(len(captions)):
                caption = captions[i]
                caption = caption.lower()
                # caption = caption.replace("[^A-Za-z]", '')
                # caption = caption.replace("\s+", ' ')
                caption = re.sub(r"[^a-zA-Z]", " ", caption)
                caption = re.sub(r"\s+", " ", caption)
                caption = caption.lower().strip()
                #
                caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
                captions[i] = caption


    def clean_captions(self):
        self.build_mapping()
        self.clean()

        return self.mapping

    def prepare_tokenizer(self):
        all_captions = []
        for key in self.mapping:
            for caption in self.mapping[key]:
                all_captions.append(caption)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        vocab_size = len(tokenizer.word_index) + 1
        max_len = max([len(caption.split()) for caption in all_captions])

        with open(self.tokenizer_file, "wb") as f:
            pickle.dump(tokenizer, f)
        
        return tokenizer, vocab_size, max_len
