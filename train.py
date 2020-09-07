import datetime
import glob
import random
import tensorflow as tf
import model
from config import Config
from utils.ImageUtils import ImageUtils
from tensorflow.keras.utils import Progbar
import time

def train():
  config = Config()
  summary_writer = tf.summary.create_file_writer(
    config.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  image_utils = ImageUtils()

  sketch_files = glob.glob(config.SKETCH_PATH + '/**/*.png', recursive=True)
  random.shuffle(sketch_files)
  num_training_samples = len(sketch_files) - 100
  train_dataset = tf.data.Dataset.from_generator(image_utils.photo_name_generator, (tf.string, tf.string), args=[sketch_files[:-100]])
  train_dataset = train_dataset.map(image_utils.load_image_train,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

  train_dataset = train_dataset.batch(config.BATCH_SIZE)

  test_dataset = tf.data.Dataset.from_generator(image_utils.photo_name_generator, (tf.string, tf.string), args=[sketch_files[-100:]])
  test_dataset = test_dataset.map(image_utils.load_image_test)
  test_dataset = test_dataset.batch(config.BATCH_SIZE)

  generator = model.Generator(config.IMG_WIDTH, config.IMG_HEIGHT, config.OUTPUT_CHANNELS)

  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  discriminator = model.Discriminator(config.IMG_WIDTH, config.IMG_HEIGHT)

  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

  @tf.function
  def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = generator(input_image, training=True)

      disc_real_output = discriminator([input_image, target], training=True)
      disc_generated_output = discriminator([input_image, gen_output], training=True)

      gen_total_loss, gen_gan_loss, gen_l1_loss = model.generator_loss(disc_generated_output, gen_output, target, config.LAMBDA)
      disc_loss = model.discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
      tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
      tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
      tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
      tf.summary.scalar('disc_loss', disc_loss, step=epoch)

    image_utils = ImageUtils()

  def fit(train_ds, epochs, test_ds, num_training_samples):
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    metrics_names = ['acc', 'pr']
    pb_i = Progbar(num_training_samples, stateful_metrics=metrics_names)
    for epoch in range(epochs):
      start = time.time()

      for example_input, example_target in test_ds.take(1):
        model.generate_images(generator, example_input, example_target)
      print("Epoch: ", epoch)

      # Train
      for n, (input_image, target) in train_ds.enumerate():
        train_step(input_image, target, epoch)
        pb_i.update(int(n + 1) * config.BATCH_SIZE)

      # saving (checkpoint) the model every 20 epochs
      if (epoch + 1) % 20 == 0:
        checkpoint.save(file_prefix=model.checkpoint_prefix)

      print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                         time.time() - start))
    checkpoint.save(file_prefix=model.checkpoint_prefix)

  fit(train_dataset, config.EPOCHS, test_dataset, num_training_samples)


if __name__ == "__main__":
  train()