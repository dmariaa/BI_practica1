import tensorflow
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, \
    Input, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16


def conv_block(input: tensorflow.Tensor, num_filters: int):
    """
    Generates a UNET encoder block including:
    CONV2D + CONV2D + MAXPOOL + RELU
    :param input: input tensor
    :param num_filters: number of filters
    :return:
    """
    x = Conv2D(num_filters=num_filters, kernel_size=3, activation='relu', padding="same",
               kernel_initializer='he_normal')(input)
    x = Conv2D(num_filters=num_filters, kernel_size=3, activation='relu', padding="same",
               kernel_initializer='he_normal')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    return x


def decoder_block_transpose(input: tensorflow.Tensor, skip_features: tensorflow.Tensor, num_filters: int):
    """
    Generates a UNET decoder block including:
    :param input: input tensor
    :param skip_features: input skip features
    :param num_filters: number of filters
    :return:
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def decoder_block_upsample(input: tensorflow.Tensor, skip_features: tensorflow.Tensor, num_filters: int):
    """
    Generates a UNET decoder block including:
    CONV2D + UPSAMPLING + CONCATENATE + CONV2D + CONV2D

    :param input: input tensor
    :param skip_features: skip features tensor
    :param num_filters: number of filters in output
    :return:
    """
    x = Conv2D(filters=num_filters, kernel_size=2, activation='relu', padding='same',
               kernel_initializer='he_normal')(input)
    x = UpSampling2D(size=(2, 2))(x)

    x = concatenate([x, skip_features], axis=3)
    x = Conv2D(filters=num_filters, kernel_size=3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = Conv2D(filters=num_filters, kernel_size=3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    return x


def build_unet(input_shape: tuple, num_classes: int = 2):
    """
    Builds unet model
    :param input_shape: input shape
    :param num_classes: number of output classes
    :return: model
    """
    # Input layer
    inputs = Input(input_shape)

    # Encoder
    e1 = conv_block(inputs, 64)
    e2 = conv_block(e1, 128)
    e3 = conv_block(e2, 256)
    e4 = conv_block(e3, 512)

    b1 = conv_block(e4, 1024)

    # Decoder - upsample2d
    d1 = decoder_block_upsample(b1, e4, 512)
    d2 = decoder_block_upsample(d1, e3, 256)
    d3 = decoder_block_upsample(d2, e2, 128)
    d4 = decoder_block_upsample(d3, e1, 128)

    # Output
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(d4)

    model = Model(inputs, outputs, name="VGG16_UNet_ear_detection")
    return model


def build_vgg16_unet(input_shape: tuple, num_classes: int = 2):
    """
    Builds UNET with vgg16 imagenet pretrained encoder
    :param input_shape: input shape
    :param num_classes: number of classes
    :return: model
    """
    # Input layer
    inputs = Input(input_shape)

    # Pretrained VGG16 encoder
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    # VGG16 Encoder layers
    s1 = vgg16.get_layer("block1_conv2").output  ## (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output  ## (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output  ## (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output  ## (64 x 64)

    b1 = vgg16.get_layer("block5_conv3").output  ## (32 x 32)

    # Decoder - transpose
    # d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    # d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    # d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    # # d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    # Decoder - upsample2d
    d1 = decoder_block_upsample(b1, s4, 512)
    d2 = decoder_block_upsample(d1, s3, 256)
    d3 = decoder_block_upsample(d2, s2, 128)
    d4 = decoder_block_upsample(d3, s1, 128)

    # Output
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(d4)

    model = Model(inputs, outputs, name="VGG16_UNet_ear_detection")
    return model


if __name__ == "__main__":
    model = build_vgg16_unet((640, 480, 3))
    model.summary()
