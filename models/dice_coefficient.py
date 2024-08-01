import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the Dice Coefficient.
    
    Args:
        y_true (tf.Tensor): Ground truth binary mask.
        y_pred (tf.Tensor): Predicted binary mask.
    
    Returns:
        tf.Tensor: Dice coefficient.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the Dice Loss, which is 1 - Dice Coefficient.
    
    Args:
        y_true (tf.Tensor): Ground truth binary mask.
        y_pred (tf.Tensor): Predicted binary mask.
    
    Returns:
        tf.Tensor: Dice loss.
    """
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate a combined loss of Binary Cross-Entropy (BCE) and Dice Loss.
    
    Args:
        y_true (tf.Tensor): Ground truth binary mask.
        y_pred (tf.Tensor): Predicted binary mask.
    
    Returns:
        tf.Tensor: Combined loss.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return bce + dl

# Custom objects for model
custom_objects = {
    'dice_coefficient': dice_coefficient,
    'combined_loss': combined_loss,
}