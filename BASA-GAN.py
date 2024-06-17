import np as np
import math
import warnings
import sio as sio
from scipy.io import loadmat
from keras.utils import to_categorical

warnings.filterwarnings('ignore')
from numpy import expand_dims, zeros, ones, asarray
from numpy.random import randn, randint, rand, random
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.layers import LeakyReLU, Dropout, Lambda, Activation, ReLU
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.python.keras.backend as K
from keras import Model, Sequential
from keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, Embedding, \
    Concatenate, multiply, Flatten
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import load_model  # 加载模型函数
# %% --------------------------------------- BAGAN--Fix Seeds -----------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = tf.keras.initializers.glorot_normal(seed=SEED)
# %% ---------------------------------- Data Preparation ---------------------------------------------------------------
def change_image_shape(images):
    shape_tuple = images.shape
    if len(shape_tuple) == 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], 1)
    elif shape_tuple == 4 and shape_tuple[-1] > 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], shape_tuple[1])
    return images
image_ind = 10
train_data = loadmat('./BSGAN-GP-zrm/SVHN/train_32x32.mat')
test_data = loadmat('./BSGAN-GP-zrm/SVHN/test_32x32.mat')

TRAIN_X = np.transpose(train_data['X'],(3,0,1,2))
TRAIN_Y = train_data['y']
x_test = np.transpose(test_data['X'],(3,0,1,2))
y_test = test_data['y']

label_zrm = TRAIN_Y
img_zrm = TRAIN_X
"""修改标签从1-10到0-9,即将10变成0标签"""
label_zrm[label_zrm == 10] = 0
y_test[y_test == 10] = 0
img_zrm = change_image_shape(img_zrm)

label_zrm = label_zrm.ravel()
y_test = y_test.ravel()#将二维标签一维数组化
channel = img_zrm.shape[-1]
print(channel)
n_classes = len(np.unique(label_zrm))
every_labels = np.unique(label_zrm)
dict_labels = {0: '10', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(11, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img_zrm[i])
    plt.xlabel(dict_labels[label_zrm[i]])
plt.pause(5)
plt.close()
#--------------Data----------------------------------------------------------------------
for c in range(1, 10):
    img_zrm_B = np.vstack(
        [img_zrm[label_zrm != c], img_zrm[label_zrm == c][:100 * c]])
    label_zrm_B = np.append(label_zrm[label_zrm != c], np.ones(100 * c) * c)
for i in range(10):
    X_with_class = img_zrm_B[label_zrm_B == i]  # get all images for this class
    print('第', i, '类样本的数量', len(X_with_class))
print(img_zrm_B.shape, label_zrm_B.shape)

# ----------------------------------------------------------------------------------
optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
d_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
g_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)

#------------------A-Embedding-------------------------------------------------------
from sklearn.model_selection import train_test_split
import cv2
# to 64 x 64 x channel通道
real = np.ndarray(shape=(img_zrm_B.shape[0], 64, 64, channel))
for i in range(img_zrm_B.shape[0]):
    real[i] = cv2.resize(img_zrm_B[i], (64, 64)).reshape((64, 64, channel))

# Train test split, for autoencoder (actually, this step is redundant if we already have test set)
x_train_A, x_test_A, y_train_A, y_test_A = train_test_split(real, label_zrm_B, test_size=0.3, shuffle=True, random_state=42)

# It is suggested to use [-1, 1] input for GAN training
x_train_A = (x_train_A.astype('float32') - 127.5) / 127.5
x_test_A = (x_test_A.astype('float32') - 127.5) / 127.5

# Get image size
img_size = x_train_A[0].shape
# %% ---------------------------------- Models Setup -------------------------------------------------------------------
latent_dim=128
# trainRatio === times(Train D) / times(Train G)
trainRatio = 5
def decoder():
    # weight initialization：
    init = RandomNormal(stddev=0.02)
    #tf.keras.initializers.RandomNormal
    noise_le = Input((latent_dim,))

    x = Dense(4*4*256)(noise_le)
    x = LeakyReLU(alpha=0.2)(x)

    ## Size: 4 x 4 x 256
    x = Reshape((4, 4, 256))(x)

    ## Size: 8 x 8 x 128
    x = Conv2DTranspose(filters=128,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    ## Size: 16 x 16 x 128
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    ## Size: 32 x 32 x 64
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    ## Size: 64 x 64 x 3
    generated = Conv2DTranspose(channel, (4, 4), strides=(2, 2), padding='same', activation='tanh', kernel_initializer=init)(x)


    generator = Model(inputs=noise_le, outputs=generated)
    return generator

#Build Encoder
def encoder():
    # weight initialization
    init = RandomNormal(stddev=0.02)

    img = Input(img_size)

    x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(img)
    # x = LayerNormalization()(x) # It is not suggested to use BN in Discriminator of WGAN
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    # 4 x 4 x 256
    feature = Flatten()(x)

    feature = Dense(latent_dim)(feature)
    out = LeakyReLU(0.2)(feature)

    model = Model(inputs=img, outputs=out)
    return model

# Build Embedding model
def embedding_labeled_latent():
    # # weight initialization
    # init = RandomNormal(stddev=0.02)

    label = Input((1,), dtype='int32')
    noise = Input((latent_dim,))
    # ne = Dense(256)(noise)
    # ne = LeakyReLU(0.2)(ne)

    le = Flatten()(Embedding(n_classes, latent_dim)(label))
    # le = Dense(256)(le)
    # le = LeakyReLU(0.2)(le)

    noise_le = multiply([noise, le])
    # noise_le = Dense(latent_dim)(noise_le)

    model = Model([noise, label], noise_le)

    return model

# Build Autoencoder
def autoencoder_trainer(encoder, decoder, embedding):

    label = Input((1,), dtype='int32')
    img = Input(img_size)

    latent = encoder(img)
    labeled_latent = embedding([latent, label])
    rec_img = decoder(labeled_latent)
    model = Model([img, label], rec_img)

    model.compile(optimizer=optimizer, loss='mae')
    return model

# Train Autoencoder
en = encoder()
de = decoder()
em = embedding_labeled_latent()
ae = autoencoder_trainer(en, de, em)

ae.fit([x_train_A, y_train_A], x_train_A,
       epochs=0,#epochs=25
       batch_size=128,
       shuffle=True,
       validation_data=([x_test_A, y_test_A], x_test_A))

# Show results of reconstructed images
decoded_imgs = ae.predict([x_test_A, y_test_A])
print(decoded_imgs.shape)
n = n_classes
plt.figure(figsize=(2*n, 4))
decoded_imgs = decoded_imgs*0.5 + 0.5
x_real_A = x_test_A*0.5 + 0.5
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    if channel == 3:
        plt.imshow(x_real_A[y_test_A==i][0].reshape(64, 64, channel))
    else:
        plt.imshow(x_real_A[y_test_A==i][0].reshape(64, 64))
        plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    if channel == 3:
        plt.imshow(decoded_imgs[y_test_A==i][0].reshape(64, 64, channel))
    else:
        plt.imshow(decoded_imgs[y_test_A==i][0].reshape(64, 64))
        plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#--------------------------------------SN-------------------------------------------
"""
Adding SN requires packaging each convolutional layer 
(as well as other layers that may require normalization, such as fully connected layers) of the discriminator
"""
class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-5, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.iteration = iteration
        self.eps = eps
        self._u = None
        self._v = None
        self.uses_learning_phase = False

    def build(self, input_shape):
        super(SpectralNormalization, self).build(input_shape)
        # 初始化u和v
        self._u = self.add_weight(
            shape=(1, self.layer.kernel.shape[0]),
            initializer='uniform',
            trainable=False,
            name='sn_u'
        )
        self._v = self.add_weight(
            shape=(self.layer.kernel_size[0] * self.layer.kernel_size[1] * self.layer.filters, 1),
            initializer='uniform',
            trainable=False,
            name='sn_v'
        )

    def call(self, inputs, training=False):
        # reshape
        w = self.layer.kernel
        w_reshape = tf.reshape(w, (-1, tf.shape(w)[-1]))

        def l2normalize(v, eps=1e-12):
            return v / (tf.reduce_sum(v**2) ** 0.5 + eps)

        u_hat = l2normalize(self._u)
        v_hat = l2normalize(self._v)

        for _ in range(self.iteration):
            v_ = l2normalize(tf.matmul(u_hat, w_reshape, transpose_b=True))
            u_ = l2normalize(tf.matmul(v_, w_reshape))
            u_hat, v_hat = u_, v_

        sigma = tf.reduce_sum(tf.matmul(tf.matmul(v_hat, w_reshape, transpose_b=True), u_hat))
        w_bar = w / sigma

        w_bar = tf.reshape(w_bar, self.layer.kernel.shape)


        self.layer.kernel.assign(w_bar)

        return self.layer(inputs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
#---------------------------------------SA--------------------------------------------
def self_attention(inputs):
    f_q = Conv2D(filters=inputs.shape[-1] // 8, kernel_size=1, padding='same')(inputs)
    f_k = Conv2D(filters=inputs.shape[-1] // 8, kernel_size=1, padding='same')(inputs)
    f_v = Conv2D(filters=inputs.shape[-1], kernel_size=1, padding='same')(inputs)
    attention_scores = tf.matmul(f_q, f_k, transpose_b=True)
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    output = tf.matmul(attention_scores, f_v)
    gamma = tf.Variable(initial_value=0.0, trainable=True)
    output = gamma * output + inputs

    return output
# define the standalone generator model
def define_generator(latent_dim):
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 256 * 4 * 4

    # 4*4*256
    X = Dense(n_nodes)(in_lat)
    X = tf.keras.layers.ReLU()(X)  #
    # X = LeakyReLU(alpha=0.2)(X)
    X = Reshape((4, 4, 256))(X)
    X = self_attention(X)


    X = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(X)
    X = tf.keras.layers.ReLU()(X)
    # X = LeakyReLU(alpha=0.2)(X)
    #X = self_attention(X)


    X = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(X)
    X = BatchNormalization()(X)
    X = tf.keras.layers.ReLU()(X)
    # X = LeakyReLU(alpha=0.2)(X)
    #X = self_attention(X)

    # 32x32x3
    out_layer = Conv2DTranspose(3, (3, 3), strides=(2, 2), activation='tanh', padding='same')(
        X)
    model = Model(in_lat, out_layer)
    return model

gen_model = define_generator(latent_dim=100)
print(gen_model.summary())

# -------------------------------------------------------------------------------
# 定义鉴别器
def define_discriminator(in_shape=(32, 32, 3), n_classes=10):
    # def define_discriminator(in_shape=(32,32,3), n_classes=10):
    in_image = Input(shape=in_shape)  # (28,28,1)

    X = SpectralNormalization(Conv2D(32, (3, 3), strides=(2, 2), padding='same'))(in_image)
    # X = LeakyReLU(alpha=0.2)(X)
    X = tf.keras.layers.ReLU()(X)
    X = self_attention(X)

    X = SpectralNormalization(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))(X)
    # X = LeakyReLU(alpha=0.2)(X)
    X = tf.keras.layers.ReLU()(X)
    #leakyRelu
    X = self_attention(X)

    X = SpectralNormalization(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))(X)
    # X = LeakyReLU(alpha=0.2)(X)
    X = tf.keras.layers.ReLU()(X)
    X = self_attention(X)

    X = Flatten()(X)
    X = Dropout(0.4)(X)  # minimize overfitting
    #X = Dense(n_classes)(X)
    X = SpectralNormalization(Dense(n_classes))(X)


    model = Model(inputs=in_image, outputs=X)

    return model

# -------------------------------------------------------------------------------
def define_sup_discriminator(disc):
    model = Sequential()
    model.add(disc)
    model.add(Activation('softmax'))
    # Let us use sparse categorical loss so we dont have to convert our Y to categorical
    model.compile(optimizer=d_optimizer,
                  loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

# -------------------------------------------------------------------------------
# 定义无监督分类器
def custom_activation(x):
    Z_x = K.sum(K.exp(x), axis=-1, keepdims=True)
    D_x = Z_x / (Z_x + 1)
    return D_x


def define_unsup_discriminator(disc):
    model = Sequential()
    model.add(disc)
    model.add(Lambda(custom_activation))
    model.compile(loss='binary_crossentropy', optimizer=d_optimizer)
    return model

disc = define_discriminator()
disc_sup = define_sup_discriminator(disc)
disc_unsup = define_unsup_discriminator(disc)
print(disc_unsup.summary())
print(disc.summary())


# -------------------------------------------------------------------------------
def define_gan(gen_model, disc_unsup):
    disc_unsup.trainable = False
    gan_output = disc_unsup(gen_model.output)
    model = Model(gen_model.input, gan_output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

gan_model = define_gan(gen_model, disc_unsup)
print(gan_model.summary())

# -------------------------------------------------------------------------------
########################################################################
def load_real_samples():
    (trainX, trainY) = (img_zrm, label_zrm)
    #X = expand_dims(trainX, axis=-1)
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5
    return [X, trainY]

def load_real_samples_B():
    (trainX, trainY) = (img_zrm_B, label_zrm_B)
    #X = expand_dims(trainX, axis=-1)
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5
    return [X, trainY]

# -------------------------------------------------------------------------------
# 选择有监督样本，取部分图片 如5000张，分类设置为10，每类选择100张
def select_supervised_samples(dataset, n_samples=5000, n_classes=10):
    X, y = dataset
    X_list, y_list = list(), list()
    n_per_class = int(n_samples / n_classes)  # Number of amples per class.
    for i in range(n_classes):
        X_with_class = X[y == i]  # get all images for this class
        # X_with_class = X[y.reshape(-1) == i]
        ix = randint(0, len(X_with_class), n_per_class)  # choose random images for each class
        [X_list.append(X_with_class[j]) for j in ix]  # add to list
        [y_list.append(i) for j in ix]
    return asarray(X_list), asarray(y_list)  # Returns a list of 2 numpy arrays corresponding to X and Y


def generate_real_samples(dataset1, n_samples):
    images, labels = dataset1
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples,1))
    return [X, labels], y

def generate_latent_points(latent_dim, n_samples):
    z_input = randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

def generate_fake(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)  #
    fake_images = generator.predict(z_input)  #
    y = zeros((n_samples, 1))  #
    return fake_images, y  #

# -------------------------------------------------------------------------------
def summarize_performance(step, gen_model, disc_sup, latent_dim, dataset, n_samples=100):
    n_classes = len(np.unique(label_zrm))  # len(np.unique(y_train))就是n_classes
    n = n_classes
    X, _ = generate_fake(gen_model, latent_dim, n_samples)
    X = (X + 1) / 2.0  # scale to [0,1] for plotting
    plt.figure(figsize=(10, 10))
    for i in range(100):
        plt.subplot(11, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i].reshape(32, 32, 3))
    filename1 = 'generated_plot_%04d.png' % (step + 1)
    plt.savefig(filename1)
    print('%04d' % (step + 1))
    plt.pause(5)
    plt.close()

    X, y = dataset
    _, acc = disc_sup.evaluate(X, y, verbose=0)
    print('Discriminator Accuracy: %.3f%%' % (acc * 100))
    # save the generator model
    filename2 = 'gen_model_%04d.h5' % (step + 1)
    gen_model.save(filename2)
    print('%04d .h5' % (step + 1))
    # Discriminator
    filename3 = 'disc_sup_%04d.h5' % (step + 1)
    disc_sup.save(filename3)
    print('%04d .h5' % (step + 1))
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))
    return acc

# --------------------------------------+GP----------------------------------------
def Gradient_Penalty(disc, half_batch, real_images, fake_images):
    """ Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.
    """
    # get the interplated image
    alpha = tf.random.normal([half_batch, 32, 32, 3], 0.0, 1.0)
    #print(fake_images.shape, real_images.shape)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # Get the discriminator output for this interpolated image.
        pred = disc(interpolated)
        # pred = self.discriminator([interpolated, labels], training=True)
    # 2. w.r.t。Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calcuate the norm of the gradients
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def train_critic(disc, X_real, X_fake, latent_dim, batch_size):
    real_images = X_real
    fake_images = X_fake

    with tf.GradientTape() as gradient_tape, tf.GradientTape() as total_tape:
        # gradient penalty
        epsilon = tf.random.uniform((batch_size, 1, 1, 1))
        # epsilon = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolates = epsilon * (real_images - fake_images) + fake_images
        gradient_tape.watch(interpolates)

        critic_interpolates = disc(interpolates)
        gradients_interpolates = gradient_tape.gradient(critic_interpolates, [interpolates])

        gradient_penalty = tf.square(gradients_interpolates)
        gradient_penalty = tf.reduce_sum(gradient_penalty, axis=np.arange(1, len(gradient_penalty.shape)))
        gradient_penalty = tf.sqrt(gradient_penalty)
        gradient_penalty = tf.reduce_mean((gradient_penalty - 1) ** 2)
        gradient_penalty = 0.005 * gradient_penalty

    return gradient_penalty


# ------------------------------------------------------------------
def train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs, n_batch):
    X_sup, y_sup = select_supervised_samples(dataset)  #
    print( X_sup.shape, y_sup.shape)
    bat_per_epo = int(dataset[0].shape[0] / n_batch)  # n_batch
    n_steps = bat_per_epo * n_epochs  # 所有步=迭代次数*周期
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, half_batch=%d, b/e=%d, steps=%d ' % (
    n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    #######################  修 改  ####################################################################
    zrm_acc_epoch = []
    zrm_loss_sup = []
    zrm_acc_sup = []
    zrm_loss_d_real = []
    zrm_loss_d_fake = []
    zrm_loss_d = []
    zrm_loss_g = []
    #  枚举周期
    for i in range(n_steps):
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], 250)
        sup_loss, sup_acc = disc_sup.train_on_batch(Xsup_real, ysup_real)

        unlabel_batch = 250
        [X_real, _], y_real = generate_real_samples(dataset, unlabel_batch)
        X_fake, y_fake = generate_fake(gen_model, latent_dim, unlabel_batch)
        d_loss_real = disc_unsup.train_on_batch(X_real, y_real)
        d_loss_fake = disc_unsup.train_on_batch(X_fake, y_fake)
        # ----------------------------------------------
        batch_size = n_batch
        # gp = train_critic(disc, X_real, X_fake, latent_dim, unlabel_batch)
        # d_loss_fake = d_loss_fake + 0.005 * gp
        # d_loss_real = d_loss_real + 0.005 * gp
        d_loss = d_loss_real + d_loss_fake
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))  # 生成规模
        gan_loss = gan_model.train_on_batch(X_gan, y_gan)
        print('>%d, [sup损失%.3f,sup准确%.3f], [d真损失%.3f,d假损失%.3f], [d损失%.3f], [g损失%.3f]' % (
            i + 1, sup_loss, sup_acc * 100, d_loss_real, d_loss_fake, d_loss,
            gan_loss))
        if i % 20 == 0:
            zrm_loss_sup.append(sup_loss)
            zrm_acc_sup.append(sup_acc)
            zrm_loss_d_real.append(d_loss_real)
            zrm_loss_d_fake.append(d_loss_fake)
            zrm_loss_d.append(d_loss)
            zrm_loss_g.append(gan_loss)
        if (i + 1) % (bat_per_epo * 1) == 0:
            print('%.0f' % i)
            zrm_acc_epoch.append(summarize_performance(i, gen_model, disc_sup, latent_dim, dataset))

    plt.figure(figsize=(7, 3))
    x = np.arange(1, len(zrm_loss_sup) + 1, 1)
    plt.plot(x, zrm_loss_sup, label="C_Loss", color='#00FF7F')
    plt.plot(x, zrm_loss_g, label="G_Loss", color='#EEAD0E')
    plt.ylabel('sup-Loss、GAN_Loss')
    plt.xlabel('Batch')
    plt.legend()
    plt.title("1- sup_Loss and GAN_Loss", fontsize=24)
    plt.figure(figsize=(7, 3))
    plt.plot(x, zrm_loss_d_real, label="D_Real", color='green')
    plt.plot(x, zrm_loss_d_fake, label="D_Fake", color='red')
    plt.ylabel('Loss of D_Real and D_Fake')
    plt.xlabel('Batch of every epoch')
    plt.legend()
    plt.title("2- Discriminator Loss During Training")

    plt.figure(figsize=(7, 3))
    x2 = np.arange(1, n_epochs + 1, 1)
    plt.plot(x2, zrm_acc_epoch, label="Every_Epoch_DAcc", color='#B03060')
    for a, b in zip(x2, zrm_acc_epoch):
        plt.text(a, b, round(100 * b, 2), ha='center', va='bottom', fontsize=7)
    plt.ylabel('Every_Epoch_DAcc')
    plt.xlabel('Epoch')
    plt.title("3- Every_Epoch_DAcc", fontsize=24)
    plt.legend()
    plt.show()
    print("最高精度", max(zrm_acc_epoch), "位置", zrm_acc_epoch.index(max(zrm_acc_epoch)))

############################绘图结束############################################
# ---------------------------------- TRAIN--------------------------------------
#
latent_dim = 100
disc = define_discriminator()
disc_sup = define_sup_discriminator(disc)
disc_unsup = define_unsup_discriminator(disc)

gen_model = define_generator(latent_dim)

gan_model = define_gan(gen_model, disc_unsup)

dataset = load_real_samples()

#dataset = load_real_samples_B()
# 注意:在本例中平衡数据集，1 epoch = 600步。不平衡数据集，1 epoch=549步
train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs=5, n_batch=100)#n_batch改成128
# 一批次100张图，一周期输入600批次。然后迭代10周期==600000万张图

##########################################训练结束################################################################
# ---------------------------------------用训练后的生成器平均生成同等数量图像-------------------
# Plot generated images
def show_plot(examples, n2):
    if n2 == 0: return print("None")
    n = math.sqrt(n2)
    n = int(n - 0.5)
    print(n, "n2=", n2)
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    plt.show()
def show_plot_fake(examples, n2):
    if n2 == 0: return print("None")
    for i in range(10 * 25):
        plt.subplot(10, 25, 1 + i)
        plt.axis('off')
        # plt.imshow(examples[i, :, :, :], cmap='gray')
        plt.imshow(examples[i])
    plt.show()
# -----------------------------------------分析评估-----------------------------------------------------------------
disc_sup_trained_model = load_model('disc_sup_18300.h5')
# 加载平衡测试数据
(testX, testy) = (x_test, y_test)
for c in range(1, 10):
    img_zrm_B_test = np.vstack(
        [testX[testy != c], testX[testy == c][:7 * c]])  # vstack()。
    label_zrm_B_test = np.append(testy[testy != c], np.ones(7 * c) * c)

for i in range(10):
    X_with_class = testX[testy == i]  # get all images for this class
    print('第', i, '类样本的数量', len(X_with_class))
# # 加载不平衡测试数据
# (testX, testy) = (img_zrm_B_test, label_zrm_B_test)
# expand to 3d, e.g. add channels
testX = expand_dims(testX, axis=-1)
# convert from ints to floats
testX = testX.astype('float32')

# scale from [0,255] to [-1,1]
testX = (testX - 127.5) / 127.5
# evaluate the model
_, test_acc = disc_sup_trained_model.evaluate(testX, testy, verbose=0)
print('Test Accuracy: %.3f%%' % (test_acc * 100))

# Predicting the Test set results
y_pred_test = disc_sup_trained_model.predict(testX)
prediction_test = np.argmax(y_pred_test, axis=1)
print(prediction_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(6, 6))
cm = confusion_matrix(testy, prediction_test)
print("cm-------------------", cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.title('Confusion matrix for:\n{}'.format('BSGAN-GP'))
plt.title('Confusion matrix')
plt.show()

# 计算F1分数
from sklearn.metrics import f1_score
f1 = f1_score(testy, prediction_test, average='macro')
print("F1 Score: %.3f" % f1)
f1_per_class = f1_score(testy, prediction_test, average=None)
print("F1 Score per class:", f1_per_class)

# -------------------每个类的测试精度------------------------------------
test_labels = to_categorical(testy)  # 真实标签
predictions = disc_sup_trained_model.predict(testX)  # 预测标签
predictionsnum = cm.diagonal()
print(predictionsnum)
Acc_i = []
for i in range(10):
    true_count = sum(test_labels[:, i] == 1)
    pred_count = predictionsnum[i]
    Acc_i.append(pred_count / true_count)
    print(f"Class {i}: Accuracy = {Acc_i[-1]:.3%}")

plt.figure(figsize=(7, 3))
x2 = np.arange(0, len(Acc_i), 1)
plt.plot(x2, Acc_i, label="Every_Class_DAcc", color='#B03060')
for a, b in zip(x2, Acc_i):
    plt.text(a, b, round(100 * b, 2), ha='center', va='bottom', fontsize=12)
plt.ylabel('Every_Class_Acc')
plt.xlabel('Epoch')
plt.title("Every_Class_Acc", fontsize=24)
plt.legend()
plt.show()

# load model
gen_trained_model = load_model('h5/SVHN/生成器仅全连接层鉴别器三层gen_model_18300.h5')  # Model trained for 100 epochs
# gen_trained_model = load_model('Mnsit-Fashion h5/gen_model_3000.h5') #Model trained for 100 epochs
# generate images
latent_points = generate_latent_points(100, 25)  # Latent dim=100 and n_samples=25
# generate images
X = gen_trained_model.predict(latent_points)
X = (X + 1) / 2.0
X = (X * 255).astype(np.uint8)
show_plot(X, 25)
