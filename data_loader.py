"""
data_loader.py
Loads soil images from class-named folders and prepares them for training.

Expected folder structure:
dataset/
├── nitrogen_deficient/
├── phosphorus_deficient/
├── potassium_deficient/
└── healthy/

Rename the folders below (CLASS_NAMES) if your actual category names differ.
"""

import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = [
    "nitrogen_deficient",
    "phosphorus_deficient",
    "potassium_deficient",
    "healthy",
]


def get_datasets(dataset_dir="dataset", validation_split=0.2, seed=123):
    """
    Loads images from dataset_dir, splits into train/validation sets,
    and applies basic data augmentation to the training set.

    Returns:
        train_ds, val_ds: tf.data.Dataset objects ready for model.fit()
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )

    # Basic data augmentation, applied only to training data
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(normalization(x), training=True), y)
    )
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    # Improve performance by prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
