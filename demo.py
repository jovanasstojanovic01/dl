import pickle
import os
import argparse

from utils import visualize_prediction
from config import Config

from tensorflow.keras.models import load_model, Model # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16 # type: ignore
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input # type: ignore

def main(image_path):
    MODEL_PATH = "working/caption_model.keras"
    TOKENIZER_PATH = "tokenizer.pkl"

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(Config.WORKING_DIR, "max_len.txt"), "r") as f:
        max_len = int(f.read().strip())

    vgg_model = InceptionV3()#VGG16()
    vgg_model = Model(inputs = vgg_model.inputs, outputs = vgg_model.layers[-2].output)

    visualize_prediction(image_path, vgg_model, model, tokenizer, max_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Pokretanje captioning modela na slici")
    parser.add_argument("image_path", type = str, help="Putanja do slike za captioning")
    args = parser.parse_args()
    main(args.image_path)