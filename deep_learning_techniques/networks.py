import tensorflow as tf

class NarrowUNet(tf.keras.Model):
    def __init__(self, name='vgg'):
     #Build the model
     inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

     #Contraction path
     c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
     c1 = tf.keras.layers.Dropout(0.1)(c1)
     c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
     p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

     c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
     c2 = tf.keras.layers.Dropout(0.1)(c2)
     c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
     p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

     c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
     c3 = tf.keras.layers.Dropout(0.2)(c3)
     c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

     #Expansive path
     u4 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
     u4 = tf.keras.layers.concatenate([u4, c2])
     c4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u4)
     c4 = tf.keras.layers.Dropout(0.1)(c4)
     c4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

     u5 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
     u5 = tf.keras.layers.concatenate([u5, c1], axis=3)
     c5 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
     c5 = tf.keras.layers.Dropout(0.1)(c5)
     c5 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

     outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='relu')(c5)

     super().__init__(inputs=inputs, outputs=outputs, name=name)
