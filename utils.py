import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore

def idx_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if index == idx:
            return word
    return None  

def predict_caption(model, image_features, tokenizer, max_len):
    in_text = "startseq"
    for _ in range(max_len):
        in_seq = tokenizer.texts_to_sequences([in_text])[0]
        in_seq = pad_sequences([in_seq], maxlen = max_len, padding = 'post')
        yhat = np.argmax(model.predict([image_features, in_seq], verbose = 0))
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        
        in_text = in_text + " " + word
        if word == "endseq":
            break

    return in_text  

def image_features_extraction(img_path, model):
    #img = load_img(img_path, target_size=(224, 224))
    img = load_img(img_path, target_size=(299, 299))  # InceptionV3 zahteva 299x299
    #
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    features = model.predict(img, verbose=0)

    return features

def visualize_prediction(img_path, vgg_model, model, tokenizer, max_len):
    features = image_features_extraction(img_path, vgg_model)
    caption = predict_caption(model, features, tokenizer, max_len)
    print("Predicted caption:", caption)
    image = Image.open(img_path)
    plt.imshow(image)
    plt.axis("off")
    plt.title(caption)
    plt.show()


#### dodaci

def predict_caption_beam_search(model, image_features, tokenizer, max_len, beam_index=3):
    start = [tokenizer.word_index['startseq']]
    sequences = [[start, 0.0]]

    while len(sequences[0][0]) < max_len:
        all_candidates = []
        for seq, score in sequences:
            padded = pad_sequences([seq], maxlen=max_len, padding='post')
            yhat = model.predict([image_features, padded], verbose=0)
            top_indices = np.argsort(yhat[0])[-beam_index:]

            for idx in top_indices:
                next_seq = seq + [idx]
                next_score = score + np.log(yhat[0][idx] + 1e-7)  # log-probabilistic scoring
                all_candidates.append([next_seq, next_score])

        # Keep beam_index best
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_index]

    final_seq = sequences[0][0]
    final_caption = [idx_to_word(idx, tokenizer) for idx in final_seq if idx_to_word(idx, tokenizer) is not None]

    final_caption = final_caption[1:]  # remove startseq
    if 'endseq' in final_caption:
        final_caption = final_caption[:final_caption.index('endseq')]

    return ' '.join(final_caption)
