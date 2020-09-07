import glob

import tensorflow as tf
import config

class ImageUtils():
  def __init__(self):
      self.config = config.Config()

  def load_png(self, image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)

    return image


  def load_jpeg(self, image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)

    return image


  def resize(self, input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


  # normalizing the images to [-1, 1]
  def normalize(self, input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

    return input_image, real_image


  def load_image_train(self, input_image, real_image):
    input_image = self.load_jpeg(input_image)
    real_image = self.load_png(real_image)
    input_image, real_image = self.resize(input_image, real_image,
                                     self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
    # input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = self.normalize(input_image, real_image)

    return input_image, real_image


  def load_image_test(self, input_image, real_image):
    input_image = self.load_jpeg(input_image)
    real_image = self.load_png(real_image)
    input_image, real_image = self.resize(input_image, real_image,
                                     self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
    input_image, real_image = self.normalize(input_image, real_image)

    return input_image, real_image

  def photo_name_generator(self, filenames):
    for png in filenames:
      jpeg = \
      glob.glob(self.config.PHOTO_PATH + '*/' + png.decode('utf8').split('\\')[-1].split('.')[0].split('-')[0] + '.jpg')[
        0].replace('\\', '/')
      yield jpeg.encode(), png
