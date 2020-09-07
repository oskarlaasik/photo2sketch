import glob
import random
import tensorflow as tf

import model
from config import Config
from utils.ImageUtils import ImageUtils

def inference():
  config = Config()
  image_utils = ImageUtils()

  sketch_files = glob.glob(config.SKETCH_PATH + '/**/*.png', recursive=True)
  random.shuffle(sketch_files)
  dataset = tf.data.Dataset.from_generator(image_utils.photo_name_generator, (tf.string, tf.string), args=[sketch_files])
  dataset = dataset.map(image_utils.load_image_test)
  dataset = dataset.batch(config.BATCH_SIZE)

  generator = model.Generator(config.IMG_WIDTH, config.IMG_HEIGHT, config.OUTPUT_CHANNELS)

  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  discriminator = model.Discriminator(config.IMG_WIDTH, config.IMG_HEIGHT)

  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

  checkpoint.restore(tf.train.latest_checkpoint(config.checkpoint_dir))

  for example_input, example_target in dataset.take(12):
    model.generate_images(generator, example_input, example_target)

if __name__ == "__main__":
  inference()