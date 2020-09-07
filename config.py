import os

class Config():
  # general config
  log_dir = "logs/"
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

  EPOCHS = 150
  LAMBDA = 100
  OUTPUT_CHANNELS = 3
  BATCH_SIZE = 12
  IMG_WIDTH = 128
  IMG_HEIGHT = 128
  SKETCH_PATH = 'data/256x256/sketch/tx_000000000000/'
  PHOTO_PATH = 'data/256x256/photo/tx_000000000000/'


