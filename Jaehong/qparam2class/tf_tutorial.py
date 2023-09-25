import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
from tf_param_fb_exporter import TfFbParamExporter

def generate_mnist():
  # Load MNIST dataset
  mnist = keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  # Normalize the input image so that each pixel value is between 0 to 1.
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  # Define the model architecture
  model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28)),
    keras.layers.Reshape(target_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
  ])

  # Train the digit classification model
  model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  model.fit(
    train_images,
    train_labels,
    epochs=1,
    validation_data=(test_images, test_labels)
  )

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  tflite_models_dir = pathlib.Path("mnist_tflite_models/")
  tflite_models_dir.mkdir(exist_ok=True, parents=True)
  tflite_model_file = tflite_models_dir / "mnist_model.tflite"
  tflite_model_file.write_bytes(tflite_model)

  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

  mnist_train, _ = tf.keras.datasets.mnist.load_data()
  images = tf.cast(mnist_train[0], tf.float32) / 255.0
  mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)

  def representative_data_gen():
    for input_value in mnist_ds.take(100):
      # Model has only one input so each data point has one element.
      yield [input_value]

  converter.representative_dataset = representative_data_gen

  tflite_16x8_model = converter.convert()
  tflite_model_16x8_file = tflite_models_dir / "mnist_model_quant_16x8.tflite"
  tflite_model_16x8_file.write_bytes(tflite_16x8_model)
  mnist_quant = TfFbParamExporter(model_content=tflite_16x8_model,
                                  json_path="out/mnist_quant/qparam.json")
  mnist_quant.save()


# if (not os.path.exists("mnist_tflite_models/mnist_model.tflite")) or (not os.path.exists("mnist_tflite_models/mnist_model_quant_16x8.tflite")):
#   generate_mnist()
generate_mnist()

# mnist = TF_FB_ParamExporter(model_path="mnist_tflite_models/mnist_model.tflite", json_path="out/mnist/qparam.json")
# mnist = TF_FB_ParamExporter(model_path="mnist_tflite_models/mnist_model.tflite", json_path="out/mnist/qparam.json")
# mnist_quant = TfFbParamExporter(model_path="mnist_tflite_models/mnist_model_quant_16x8.tflite", json_path="out/mnist_quant/qparam.json")
# mnist_quant.save()