import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", required=True, type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

model = tf.keras.models.load_model('./models/narrow-unet-unstretched1612692792')

X_predict = model.predict(
    X_test, batch_size=None, verbose=1, steps=None, max_queue_size=10,
    workers=1, use_multiprocessing=False
)
