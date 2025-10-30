import os
import tensorflow as tf
from config import Config
from tqdm import tqdm
import pickle

from feature_extraction import FeatureExtractor
from caption_processor import CaptionProcessor
from data_generator import DataGenerator
from model_builder import CaptionModelBuilder
from utils import predict_caption

from nltk.translate.bleu_score import corpus_bleu # type: ignore


def main():
    extractor = FeatureExtractor(
        os.path.join(Config.WORKING_DIR, "features.pkl"),
        os.path.join(Config.BASE_DIR, "Images")
    )
    features = extractor.extract()

    processor = CaptionProcessor(
        os.path.join(Config.BASE_DIR, "captions.txt"),
        os.path.join(Config.WORKING_DIR, "mapping.pkl"),
        os.path.join(Config.WORKING_DIR, "tokenizer.pkl")
    )
    mapping = processor.clean_captions()
    tokenizer, vocab_size, max_len = processor.prepare_tokenizer()
    
    with open(os.path.join(Config.WORKING_DIR, "max_len.txt"), "w") as f:
        f.write(str(max_len))

    # Train/test split
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.9)
    train_ids = image_ids[:split]
    test_ids = image_ids[split:]

    test_ids_path = os.path.join(Config.WORKING_DIR, "test_ids.pkl")
    with open(test_ids_path, "wb") as f:
        pickle.dump(test_ids, f)

    data_gen = DataGenerator(mapping, features, tokenizer, max_len, vocab_size, Config.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        lambda: data_gen.generator(train_ids),
        output_signature=(
            (
                tf.TensorSpec(shape = (None, 4096), dtype = tf.float32),  # X1
                tf.TensorSpec(shape = (None, max_len), dtype = tf.int32)  # X2
            ),
            tf.TensorSpec(shape = (None, vocab_size), dtype = tf.float32)  # Y
        )
    )

    builder = CaptionModelBuilder(vocab_size, max_len)
    model = builder.build()
    model.fit(dataset, epochs = Config.EPOCHS, steps_per_epoch = len(train_ids) // Config.BATCH_SIZE, verbose = 1)
    model.save(os.path.join(Config.MODELS_DIR, "caption_model.keras"))

    # test
    actual,predicted=list(), list()
    for key in tqdm(test_ids):
        captions = mapping[key]
        y_pred = predict_caption(model, features[key], tokenizer, max_len)
        
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        actual.append(actual_captions)
        predicted.append(y_pred)

    print("BLEU-1: %f"% corpus_bleu(actual, predicted, weights = (1.0,0,0,0)))
    print("BLEU-2: %f"% corpus_bleu(actual, predicted, weights = (0.5,0.5,0,0)))

def main_with_checkpoint():
    extractor = FeatureExtractor(
        os.path.join(Config.WORKING_DIR, "features.pkl"),
        os.path.join(Config.BASE_DIR, "Images")
    )
    features = extractor.extract()

    processor = CaptionProcessor(
        os.path.join(Config.BASE_DIR, "captions.txt"),
        os.path.join(Config.WORKING_DIR, "mapping.pkl"),
        os.path.join(Config.WORKING_DIR, "tokenizer.pkl")
    )
    mapping = processor.clean_captions()
    tokenizer, vocab_size, max_len = processor.prepare_tokenizer()
    
    with open(os.path.join(Config.WORKING_DIR, "max_len.txt"), "w") as f:
        f.write(str(max_len))

    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.9)
    train_ids = image_ids[:split]
    test_ids = image_ids[split:]

    val_split = int(len(train_ids) * 0.9)
    val_ids = train_ids[val_split:]
    train_ids = train_ids[:val_split]

    data_gen = DataGenerator(mapping, features, tokenizer, max_len, vocab_size, Config.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        lambda: data_gen.generator(train_ids),
        output_signature=(
            (
                tf.TensorSpec(shape = (None, 4096), dtype = tf.float32),  # X1
                tf.TensorSpec(shape = (None, max_len), dtype = tf.int32)  # X2
            ),
            tf.TensorSpec(shape = (None, vocab_size), dtype = tf.float32)  # Y
        )
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_gen.generator(val_ids),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 4096), dtype=tf.float32),  # X1
                tf.TensorSpec(shape=(None, max_len), dtype=tf.int32)  # X2
            ),
            tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)  # Y
        )
    )

    builder = CaptionModelBuilder(vocab_size, max_len)
    model = builder.build()

    checkpoint_path = os.path.join(Config.MODELS_DIR, "caption_model_checkpoint.keras")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',       
        save_best_only=True,      
        mode='min',               
        verbose=1
    )

    model.fit(
        dataset,
        validation_data=val_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_ids) // Config.BATCH_SIZE,
        validation_steps=len(val_ids) // Config.BATCH_SIZE,
        callbacks=[checkpoint],
        verbose=1
    )

    best_model = tf.keras.models.load_model(checkpoint_path)
    # test
    actual,predicted=list(), list()
    for key in tqdm(test_ids):
        captions = mapping[key]
        y_pred = predict_caption(best_model, features[key], tokenizer, max_len)
        
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        actual.append(actual_captions)
        predicted.append(y_pred)

    print("BLEU-1: %f"% corpus_bleu(actual, predicted, weights = (1.0,0,0,0)))
    print("BLEU-2: %f"% corpus_bleu(actual, predicted, weights = (0.5,0.5,0,0)))

def main_with_checkpoint_and_early_stopping():
    extractor = FeatureExtractor(
        os.path.join(Config.WORKING_DIR, "features.pkl"),
        os.path.join(Config.BASE_DIR, "Images")
    )
    features = extractor.extract()

    processor = CaptionProcessor(
        os.path.join(Config.BASE_DIR, "captions.txt"),
        os.path.join(Config.WORKING_DIR, "mapping.pkl"),
        os.path.join(Config.WORKING_DIR, "tokenizer.pkl")
    )
    mapping = processor.clean_captions()
    tokenizer, vocab_size, max_len = processor.prepare_tokenizer()
    
    with open(os.path.join(Config.WORKING_DIR, "max_len.txt"), "w") as f:
        f.write(str(max_len))

    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.9)
    train_ids = image_ids[:split]
    test_ids = image_ids[split:]

    val_split = int(len(train_ids) * 0.9)
    val_ids = train_ids[val_split:]
    train_ids = train_ids[:val_split]

    data_gen = DataGenerator(mapping, features, tokenizer, max_len, vocab_size, Config.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        lambda: data_gen.generator(train_ids),
        output_signature=(
            (
                tf.TensorSpec(shape = (None, 4096), dtype = tf.float32),  # X1
                tf.TensorSpec(shape = (None, max_len), dtype = tf.int32)  # X2
            ),
            tf.TensorSpec(shape = (None, vocab_size), dtype = tf.float32)  # Y
        )
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_gen.generator(val_ids),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 4096), dtype=tf.float32),  # X1
                tf.TensorSpec(shape=(None, max_len), dtype=tf.int32)  # X2
            ),
            tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)  # Y
        )
    )

    builder = CaptionModelBuilder(vocab_size, max_len)
    model = builder.build()

    checkpoint_path = os.path.join(Config.MODELS_DIR, "caption_model_checkpoint_early_stopping.keras")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',       
        save_best_only=True,      
        mode='min',               
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',    
        patience=5,             
        restore_best_weights=True,  
        verbose=1
    )

    model.fit(
        dataset,
        validation_data=val_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_ids) // Config.BATCH_SIZE,
        validation_steps=len(val_ids) // Config.BATCH_SIZE,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    best_model = tf.keras.models.load_model(checkpoint_path)
    # test
    actual,predicted=list(), list()
    for key in tqdm(test_ids):
        captions = mapping[key]
        y_pred = predict_caption(best_model, features[key], tokenizer, max_len)
        
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        actual.append(actual_captions)
        predicted.append(y_pred)

    print("BLEU-1: %f"% corpus_bleu(actual, predicted, weights = (1.0,0,0,0)))
    print("BLEU-2: %f"% corpus_bleu(actual, predicted, weights = (0.5,0.5,0,0)))

if __name__ == "__main__":
    # main()
    main_with_checkpoint()
    # main_with_checkpoint_and_early_stopping()
