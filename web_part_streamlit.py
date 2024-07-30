import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
import keras
import os
import time
import tensorflow as tf
import glob
import PIL
from keras.layers import Conv2DTranspose
from IPython import display
import io
# Generator

def make_generator():
    model = tf.keras.Sequential()

    model.add(layers.Dense(4 * 4 * 256, input_shape=(100,)))
    model.add(layers.Reshape((4, 4, 256)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh'))

    return model

generator = make_generator()

generator.load_weights('D:\\DCGAN\\models\\generator_weights.h5')

# predictions = generator(tf.random.normal([16, 100]), training=False)

# plt.figure(figsize=(10, 10))
# for i in range(predictions.shape[0]):
#     plt.subplot(4, 4, i+1)
#     img_array = (predictions[i].numpy() * 127.5 + 127.5).astype(np.uint8)
#     plt.imshow(img_array)
#     plt.axis('off')
# plt.show()


# Discriminator

def make_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[32, 32, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator()

discriminator.load_weights('D:\\DCGAN\\models\\discriminator_weights.h5')

# generated_image = generator(tf.random.normal([1, 100]), training=False)
# img_array = (generated_image[0].numpy() * 127.5 + 127.5).astype(np.uint8)
# plt.imshow(img_array)
# plt.axis('off')
# plt.show()
# discriminator(generated_image, training=False)

st.title("Завантаження зображення та опрацювання його моделлю")
st.write("Завантажте зображення та модель згенерує фейкове зображення та зробить висновок")

uploaded_file = st.file_uploader("Виберіть зображення...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Завантажене зображення", use_column_width=True)
    st.write("")
    st.write("Генерую зображення...")

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    img_array = (generated_image[0].numpy() * 127.5 + 127.5).astype(np.uint8)


    st.image(img_array, caption="Згенероване зображення.", use_column_width=True)


    verdict = discriminator(generated_image, training=False).numpy()[0][0]
    st.write(f"Відповідь дискримінатора: {verdict}")


    if verdict > 0:
        st.write("Дискримінатор визначив згенероване зображення правдивим.")
    else:
        st.write("Дискримінатор визначив згенероване зображення фейковим.")