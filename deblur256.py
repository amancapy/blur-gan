import time
import numpy as np
import matplotlib.pyplot as plt
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def load(img_file):
    img = tf.io.read_file(img_file)
    img = tf.io.decode_jpeg(img)

    w = tf.shape(img)[1]
    w = w // 2

    inp_img = img[:, w:, :]
    inp_img = tf.cast(inp_img, tf.float32)
    inp_img = (inp_img / 127.5) - 1
    inp_img = np.expand_dims(inp_img, 0)
    return inp_img


generator0 = tf.keras.models.load_model("good_models\\0", compile=False)
generator1 = tf.keras.models.load_model("good_models\\1", compile=False)
generator2 = tf.keras.models.load_model("good_models\\2", compile=False)

for pick in range(0, 7288, 100):
    inp = load(f"C:\\Users\\Aman\\PycharmProjects\\set-gen\\blurs\\test\\{pick}.jpeg")

    stime = time.perf_counter()
    gen0 = generator0(inp).numpy()[0]
    gen1 = generator1(inp).numpy()[0]
    gen2 = generator2(inp).numpy()[0]
    print(time.perf_counter() - stime)

    fin = np.zeros(shape=(256, 256, 3))
    for i in range(len(fin)):
        for j in range(len(fin[0])):
            for k in range(len(fin[0][0])):
                # geometric mean
                gmpix = np.cbrt(gen0[i][j][k] * gen1[i][j][k] * gen2[i][j][k])
                # [-1, 1] to [0, 1]
                gmpix = ((((gmpix - (-1)) * (1 - 0)) / (1 - (-1))) + 0)

                # arithmetic mean
                ampix = np.mean([gen0[i][j][k], gen1[i][j][k], gen2[i][j][k]])
                # [-1, 1] to [0, 1]
                ampix = ((((ampix - (-1)) * (1 - 0)) / (1 - (-1))) + 0)

                # mean of gm and am
                fin[i][j][k] = (gmpix + ampix) / 2

    plt.imsave(f"sample_gens\\{pick}.png", fin)
