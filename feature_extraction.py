import os
import pickle
from tqdm import tqdm

from utils import image_features_extraction

# from tensorflow.keras.applications.vgg16 import VGG16 # type: ignore
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input # type: ignore

from tensorflow.keras.models import Model # type: ignore

class FeatureExtractor:
    def __init__(self, features_file, images_directory):
        # vgg_model = VGG16()
        # self.model = Model(inputs = vgg_model.inputs, outputs = vgg_model.layers[-2].output)
        model = InceptionV3(weights='imagenet')
        model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
        self.model = model
        #
        self.features_file = features_file
        self.images_directory = images_directory

    def extract(self):
        features = {}
        for img_name in tqdm(os.listdir(self.images_directory)):
            img_id = img_name.split('.')[0]
            img_path = os.path.join(self.images_directory, img_name)
            features[img_id] = image_features_extraction(img_path, self.model)

        with open(self.features_file, "wb") as f:
            pickle.dump(features, f)

        with open(self.features_file, "rb") as f:
            features = pickle.load(f)
            
        return features