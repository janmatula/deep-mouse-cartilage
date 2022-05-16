#author Jan Matula
#created on 30/03/22

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, add, Conv2DTranspose, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

#parameters for Tversky loss - alpha, beta = 0.5 equals to Dice loss
alpha = 0.5
beta = 0.5

def residual_block_down(y, nb_channels):
    """
    Downsampling residual block
    y -- input
    nb_channels -- number of output channels
    """
    nb_channels = int(nb_channels)
    
    shortcut = MaxPooling2D((2, 2), padding='same')(y)
    shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1), activation='linear', padding='same', kernel_initializer="lecun_normal")(shortcut)

    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), activation='selu', padding='same', kernel_initializer="lecun_normal")(y)
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(2, 2), activation='linear', padding='same', kernel_initializer="lecun_normal")(y)
    y = add([shortcut, y])
    y = Activation("selu")(y)
    return y
def residual_block_up(y, nb_channels):
    """
    Upsampling residual block
    y -- input
    nb_channels -- number of output channels
    """
    nb_channels = int(nb_channels)

    shortcut = UpSampling2D((2, 2))(y)
    shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1), activation='linear', padding='same', kernel_initializer="lecun_normal")(shortcut)

    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), activation='selu', padding='same', kernel_initializer="lecun_normal")(y)
    y = Conv2DTranspose(nb_channels, kernel_size=(3, 3), strides=(2, 2), activation='linear', padding='same',kernel_initializer="lecun_normal")(y)
    y = add([shortcut, y])
    y = Activation("selu")(y)
    return y

def residual_block_flat(y, nb_channels):
    """
    Flat residual block
    y -- input
    nb_channels -- number of output channels
    """
    nb_channels = int(nb_channels)
    shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1), activation='linear', padding='same', kernel_initializer="lecun_normal")(y)
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), activation='selu', padding='same', kernel_initializer="lecun_normal")(y)
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), activation='linear', padding='same', kernel_initializer="lecun_normal")(y)
    y = add([shortcut, y])
    y = Activation("selu")(y)
    return y


def Tversky(ground_truth, predicted):
    """
    Tversky metrix
    when alpha=beta=0.5 equal to Dice loss
    """
    smooth=1
    ground_truth_flat = K.flatten(ground_truth)
    predicted_flat = K.flatten(predicted)
    intersection = K.sum(ground_truth_flat * predicted_flat)
    gt_pred = alpha*K.sum((1-ground_truth_flat) * predicted_flat)
    pred_gt = beta*K.sum(ground_truth_flat * (1-predicted_flat))
    return (intersection + smooth )/(intersection + smooth + gt_pred + pred_gt)
#Tversky loss
def Tversky_loss(ground_truth, predicted):
    """Tversky loss"""
    return -Tversky(ground_truth, predicted)


def residual_selu_deeplysup(image_shape, n_first_layer):
    """
    The CNN architecture
    image_shape -- the input image shape in the format, tuple (height, width, channels)
    n_first_layer -- the number of blocks in the first layer of the CNN
    """
    input_img = Input(shape=(image_shape[0], image_shape[1], image_shape[2]))
    input_layer = Conv2D(int(n_first_layer), (3, 3), activation='selu', padding='same', kernel_initializer="lecun_normal")(input_img)

    d1 = residual_block_down(input_layer, n_first_layer*2)
    d2 = residual_block_down(d1, n_first_layer*4)
    d3 = residual_block_down(d2, n_first_layer*8)
    d4 = residual_block_down(d3, n_first_layer*16)
    d5 = residual_block_down(d4, n_first_layer*32)
    d6 = residual_block_down(d5, n_first_layer*64)
    
    encoded = residual_block_flat(d6, n_first_layer*64)
    out6 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same', kernel_initializer="lecun_normal")(encoded)
    out6 = UpSampling2D((64, 64))(out6)
    u6 = residual_block_up(encoded, n_first_layer*32)

    u5 = concatenate([u6, d5], axis = - 1)
    out5 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same', kernel_initializer="lecun_normal")(u5)
    out5 = UpSampling2D((32, 32))(out5)
    u5 = residual_block_up(u5, n_first_layer*16)

    u4 = concatenate([u5, d4], axis = - 1)
    out4 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same', kernel_initializer="lecun_normal")(u4)
    out4 = UpSampling2D((16, 16))(out4)
    u4 = residual_block_up(u4, n_first_layer*8)

    u3 = concatenate([u4, d3], axis = - 1)
    out3 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same', kernel_initializer="lecun_normal")(u3)
    out3 = UpSampling2D((8, 8))(out3)
    u3 = residual_block_up(u3, n_first_layer*4)

    u2 = concatenate([u3, d2], axis=-1)
    out2 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same', kernel_initializer="lecun_normal")(u2)
    out2 = UpSampling2D((4, 4))(out2)
    u2 = residual_block_up(u2, n_first_layer*2)

    u1 = concatenate([u2, d1], axis = -1)
    out1 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same', kernel_initializer="lecun_normal")(u1)
    out1 = UpSampling2D((2, 2))(out1)
    u1 = residual_block_up(u1, n_first_layer)

    output = concatenate([u1, input_layer], axis = -1)
    output = residual_block_flat(output, n_first_layer)
    output = residual_block_flat(output, n_first_layer)
    output = Conv2D(image_shape[2], (3, 3), activation='sigmoid', padding='same')(output)
    model = Model(input_img, [out6, out5, out4, out3, out2, out1, output])

    model.compile(optimizer = Adam(lr = 1e-4, amsgrad = True), loss = [Tversky_loss, Tversky_loss, Tversky_loss, Tversky_loss, Tversky_loss, Tversky_loss, Tversky_loss],  loss_weights = [.03, .05, .08, .12, .15, .2, .37], metrics = [Tversky])

    return(model)