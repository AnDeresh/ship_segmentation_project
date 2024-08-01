from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate # type: ignore
from typing import Tuple

def conv_block(input_tensor, num_filters: int) -> Model:
    """Convolutional block consisting of two Conv2D layers."""
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    return x

def downsampling_block(input_tensor, num_filters: int) -> Tuple[Model, Model]:
    """Downsampling block consisting of a Conv2D block followed by MaxPooling2D."""
    conv = conv_block(input_tensor, num_filters)
    pool = MaxPooling2D((2, 2))(conv)
    return conv, pool

def upsampling_block(input_tensor, concat_tensor, num_filters: int) -> Model:
    """Upsampling block consisting of UpSampling2D followed by concatenation and Conv2D block."""
    upsample = UpSampling2D((2, 2))(input_tensor)
    upsample = concatenate([upsample, concat_tensor])
    conv = conv_block(upsample, num_filters)
    return conv

def unet(input_size: Tuple[int, int, int] = (128, 128, 3)) -> Model:
    """Builds the U-Net model."""
    inputs = Input(input_size)
    
    # Downsampling
    c1, p1 = downsampling_block(inputs, 64)
    c2, p2 = downsampling_block(p1, 128)
    c3, p3 = downsampling_block(p2, 256)
    c4, p4 = downsampling_block(p3, 512)
    
    # Bottleneck
    c5 = conv_block(p4, 1024)
    
    # Upsampling
    c6 = upsampling_block(c5, c4, 512)
    c7 = upsampling_block(c6, c3, 256)
    c8 = upsampling_block(c7, c2, 128)
    c9 = upsampling_block(c8, c1, 64)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model