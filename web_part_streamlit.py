import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from PIL import Image

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

generator.load_weights('./models/generator_weights.h5')


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

discriminator.load_weights('./models/discriminator_weights.h5')

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
    real_image = Image.open(uploaded_file)
    st.image(real_image, caption="Завантажене зображення", use_column_width=True)


    real_image = real_image.resize((32,32))
    real_img_array = np.array(real_image)
    real_img_array = (real_img_array - 127.5) / 127.5
    real_img_array = np.expand_dims(real_img_array, axis=0)

    verdict_real = discriminator(real_img_array, training=False).numpy()[0][0]

    st.write(f"Відповідь дискримінатора для реального зображення: {verdict_real}")
    if verdict_real > 0:
         st.write("Дискримінатор визначив реальне зображення правдивим.")
    else:
        st.write("Дискримінатор визначив реальне зображення фейковим.")

    st.write("")

    if st.button('Генерувати зображення'):
        st.write("Генерую зображення...")

        noise = tf.random.normal([1, 100])
        generated_image = generator(noise, training=False)
        generated_img_array = (generated_image[0].numpy() * 127.5 + 127.5).astype(np.uint8)

        
        st.image(generated_img_array, caption="Згенероване зображення.", use_column_width=True)

        
        verdict_fake = discriminator(generated_image, training=False).numpy()[0][0]
        
        st.write(f"Відповідь дискримінатора для згенерованого зображення: {verdict_fake}")
        if verdict_fake > 0:
            st.write("Дискримінатор визначив згенероване зображення правдивим.")
        else:
            st.write("Дискримінатор визначив згенероване зображення фейковим.")