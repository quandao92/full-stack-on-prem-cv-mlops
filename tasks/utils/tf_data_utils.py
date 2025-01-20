import pandas as pd
import tensorflow as tf
import imgaug.augmenters as iaa
from tensorflow.data import AUTOTUNE
from typing import List

augment_config = [
            iaa.Sometimes(0.5, 
                iaa.AddToBrightness((-30, 30))),
            iaa.OneOf([
                iaa.Fliplr(1.0)
            ]),
            iaa.OneOf([
                iaa.Affine(rotate=(-20, 20)),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
            ])
        ]

augmenter = iaa.Sequential(augment_config, random_order=True)
AUGMENTER = iaa.Sometimes(0.8, augmenter)

def load_data(path, label, target_size):
    # NOTE: normalization with /255 is done after augmentation cuz imgaug lib needs image to be uint8
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, target_size)
    label = tf.cast(label, tf.int32)
    return image, label

def normalize(images, label):
    result = tf.cast(images, tf.float32)
    result = result / 255.0
    return result, label

def augment(x, y, augmenter):
    # imgaug require uint8 as input
    x = tf.cast(x, tf.uint8)
    # augment_image for a single image
    # augment_imageS for a batch of image
    x = tf.numpy_function(augmenter.augment_image,
                           [x],
                           x.dtype)
    return x, y


# Functions for Time Series Data Pipeline
def load_time_series(data: tf.Tensor, label: tf.Tensor):
    data = tf.cast(data, tf.float32)
    label = tf.cast(label, tf.float32)
    return data, label

def normalize_time_series(data, label):
    data = (data - tf.reduce_min(data)) / (tf.reduce_max(data) - tf.reduce_min(data))
    return data, label


# Unified Data Pipeline for Image and Time Series
def build_data_pipeline(annot_df: pd.DataFrame, classes: List[str], split: str, data_type: str, 
                        img_size: List[int] = None, time_step: int = None, batch_size: int = 8, 
                        do_augment: bool = False, augmenter: iaa = None):
    """
    Unified data pipeline for both image and time series data.
    - `data_type`: 'image' or 'timeseries'.
    """
    df = annot_df[annot_df['split'] == split]
    
    if data_type == 'image':
        # Image pipeline
        path = df['abs_path']
        label = df[classes]
        pipeline = tf.data.Dataset.from_tensor_slices((path, label))
        pipeline = (pipeline
                    .shuffle(len(df))
                    .map(lambda path, label: load_data(path, label, target_size=img_size), num_parallel_calls=AUTOTUNE))
        
        if do_augment and augmenter:
            pipeline = pipeline.map(lambda x, y: augment(x, y, augmenter), num_parallel_calls=AUTOTUNE)
        
        pipeline = (pipeline
                    .map(normalize, num_parallel_calls=AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(AUTOTUNE))
    
    elif data_type == 'timeseries':
        # Time series pipeline
        sequences = df['sequence'].apply(lambda x: tf.convert_to_tensor(x)).values
        labels = df['label'].apply(lambda x: tf.convert_to_tensor(x)).values
        
        pipeline = tf.data.Dataset.from_tensor_slices((sequences, labels))
        pipeline = (pipeline
                    .shuffle(len(df))
                    .map(load_time_series, num_parallel_calls=AUTOTUNE)
                    .map(normalize_time_series, num_parallel_calls=AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(AUTOTUNE))
    
    else:
        raise ValueError(f"Unsupported data_type: {data_type}. Choose 'image' or 'timeseries'.")
    
    return pipeline