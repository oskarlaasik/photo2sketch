import os

class Config():
  # general config
  log_dir = "logs/"
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

  #training
  EPOCHS = 150
  LAMBDA = 100
  BATCH_SIZE = 8

  #image config
  OUTPUT_CHANNELS = 3
  IMG_WIDTH = 256
  IMG_HEIGHT = 256

  #dataset
  SKETCH_PATH = 'data/256x256/sketch/tx_000000000000/'
  PHOTO_PATH = 'data/256x256/photo/tx_000000000000/'


