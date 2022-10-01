import os
import numpy
import time
import datetime
import matplotlib.pyplot as plt
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

PATH = "C:/Users/Aman/PycharmProjects/set-gen/blurs"
sample = tf.io.read_file(str(PATH + "/train/1.jpeg"))
sample = tf.io.decode_jpeg(sample)

def load(img_file):
    img = tf.io.read_file(img_file)
    img = tf.io.decode_jpeg(img)

    w = tf.shape(img)[1]
    w = w // 2

    inp_img = img[:, w:, :]
    tar_img = img[:, :w, :]

    inp_img = tf.cast(inp_img, tf.float32)
    tar_img = tf.cast(tar_img, tf.float32)

    return inp_img, tar_img

inp, tar = load(PATH + "/train/1250.jpeg")

BUFFER_SIZE = 2500
BATCH_SIZE = 1
IMG_H, IMG_W = 256, 256

def resize(inp_img, tar_img, height, width):
    inp_img = tf.image.resize(inp_img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    tar_img = tf.image.resize(tar_img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return inp_img, tar_img

def random_crop(inp_img, tar_img):
  stacked_image = tf.stack([inp_img, tar_img], axis=0)
  cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_H, IMG_W, 3])

  return cropped_image[0], cropped_image[1]


def normalize(inp_img, tar_img):
    inp_img, tar_img = (inp_img / 127.5) - 1, (tar_img / 127.5) - 1

    return inp_img, tar_img

@tf.function()
def random_jitter(inp_img, tar_img):
    inp_img, tar_img = resize(inp_img, tar_img, 286, 286)
    inp_img, tar_img = random_crop(inp_img, tar_img)

    if tf.random.uniform(()) > 0.5:
        inp_img = tf.image.flip_left_right(inp_img)
        tar_img = tf.image.flip_left_right(tar_img)

    return inp_img, tar_img


def load_image_train(img_file):
    inp_img, tar_img = load(img_file)
    inp_img, tar_img = random_jitter(inp_img, tar_img)
    inp_img, tar_img = normalize(inp_img, tar_img)

    return inp_img, tar_img

def load_image_test(img_file):
    inp_img, tar_img = load(img_file)
    inp_img, tar_img = resize(inp_img, tar_img, IMG_H, IMG_W)
    inp_img, tar_img = normalize(inp_img, tar_img)

    return inp_img, tar_img

train_ds = tf.data.Dataset.list_files(PATH + "/train/*.jpeg")
train_ds = train_ds.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BUFFER_SIZE)
train_ds = train_ds.batch(BATCH_SIZE)

test_ds = tf.data.Dataset.list_files(PATH + "/test/*.jpeg")
test_ds = test_ds.map(load_image_test)
test_ds = test_ds.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding="same", kernel_initializer=initializer,
                                      use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.Dropout(0.25))
    result.add(tf.keras.layers.LeakyReLU())

    return result

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))

def upsample(filters, size, apply_droput=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding="same",
                                               kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_droput:
        result.add(tf.keras.layers.Dropout(0.55))
    result.add(tf.keras.layers.ReLU())

    return result

up_model = upsample(3, 4)
up_result = up_model(down_result)

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(512, 4, apply_batchnorm=False),
        downsample(256, 4),
        downsample(128, 4),
        downsample(64, 4),
        downsample(64, 4),
        downsample(64, 4),
        downsample(64, 4),
        downsample(64, 4),
    ]

    up_stack = [
        upsample(64, 4, apply_droput=True),
        upsample(64, 4, apply_droput=True),
        upsample(64, 4, apply_droput=True),
        upsample(64, 4),
        upsample(64, 4),
        upsample(128, 4),
        upsample(256, 4),
        upsample(512, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding="same",
                                           kernel_initializer=initializer, activation="tanh")

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
# gen_output = generator(inp[tf.newaxis, ...], training=False)
tf.keras.utils.plot_model(generator, to_file="generator.png", show_shapes=True, dpi=256)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = 2 * gan_loss + 0.5 * l1_loss

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.Input(shape=[256, 256, 3], name="inp_img")
    tar = tf.keras.Input(shape=[256, 256, 3], name="tar_img")

    x = tf.keras.layers.concatenate([inp, tar])

    d1 = downsample(64, 4, False)(x)
    d2 = downsample(128, 4)(d1)
    d3 = downsample(256, 4)(d2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(d3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    bnorm1 = tf.keras.layers.BatchNormalization()(conv)
    lr = tf.keras.layers.LeakyReLU()(bnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(lr)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, to_file="discriminator.png", show_shapes=True, dpi=64)
# disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
# plt.imshow(disc_out[0, ..., -1], vmin=20, vmax=20, cmap="RdBu_r")
# plt.colorbar()


def discriminator_loss(real_outut, gen_output):
    real_loss = loss_object(tf.ones_like(real_outut), real_outut)
    gen_loss = loss_object(tf.zeros_like(gen_output), gen_output)
    total_loss = real_loss + gen_loss

    return total_loss


gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

chkpt_dir = "C:\\Users\\Aman\\Desktop\\good_chkp"
ckkpt_prefix = os.path.join(chkpt_dir, "ckpt")
chkpt = tf.train.Checkpoint(gen_optimizer=gen_optimizer,
                            disc_optimizer=disc_optimizer,
                            generator=generator,
                            discriminator=discriminator)


def generate_images(model, test_input, tar, step, test=False):
    prediction = model(test_input, training=(False if test else True))
    plt.figure(figsize=(12, 4))

    display_list = [test_input[0], tar[0], prediction[0]]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(display_list[i])
        plt.axis("off")

    if not test:
        savepath = "gens"
    else:
        savepath = "testgens"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(f"{savepath}/{step}")
    plt.close()
    # plt.show()

log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(inp_image, tar_image, step, l1_fac):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(inp_image, training=True)

        disc_real_output = discriminator([inp_image, tar_image], training=True)
        disc_gen_output = discriminator([inp_image, gen_output], training=True)

        disc_loss = discriminator_loss(disc_real_output, disc_gen_output)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_gen_output, gen_output, tar)
        gen_total_loss += l1_fac * gen_l1_loss

        gen_grads = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step//1000)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step//1000)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step//1000)
            tf.summary.scalar("disc_loss", disc_loss, step=step//1000)

def fit(train_ds, test_ds, steps):
    ex_input, ex_output = next(iter(test_ds.take(1)))
    stime = time.time()

    l1_fac = 0.175
    for step, (inp_img, tar_img) in train_ds.repeat().take(steps).enumerate():
        if not step % 100:
            print(f"Time taken for 100 steps: {int(time.time()-stime)}s")
            stime = time.time()
            generate_images(generator, ex_input, ex_output, step)
            print(f"Step: {step}")
            l1_fac += 0.175

        train_step(inp_img, tar_img, step, l1_fac)

        if not (step+1) % 10:
            print(".", end="", flush=True)

        if not (step+1) % 100:
            chkpt.save(file_prefix=ckkpt_prefix)


# print(chkpt.restore(tf.train.latest_checkpoint(chkpt_dir)))
# print(generator.save("C:\\Users\\Aman\\Desktop\\good_models\\4"))
# fit(train_ds, test_ds, steps=125000)

# nice work so far keep it up ganny ^_^

# for i in range(10):
#     ex_input, ex_output = next(iter(test_ds.take(1)))
#     generate_images(generator, ex_input, ex_output, i, test=True)
