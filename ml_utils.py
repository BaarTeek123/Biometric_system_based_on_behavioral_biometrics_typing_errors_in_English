import tensorflow as tf
import numpy as np
from decorators import log_info
from mlxtend.evaluate import paired_ttest_5x2cv

@log_info
def compute_class_weights(y):
    # Count the number of occurrences of each class
    unique_classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))

    # Compute the total number of samples
    total_samples = len(y)

    # Compute weights as the inverse of each class's frequency
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    return class_weights


@log_info
def create_weighted_dataset(X, y, class_weights):
    # Assign weights based on the class of each sample
    sample_weights = np.array([class_weights[cls] for cls in y])

    # Convert to tensors
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
    weights_tensor = tf.convert_to_tensor(sample_weights, dtype=tf.float32)

    # Create a dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor, weights_tensor)).batch(1000)

    return dataset

