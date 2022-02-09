import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import load_dataset
from Model import Generator
from Model import Discriminator
#from Model import CycleGan


#   GPU SETUP
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()


#   Set DATASET_PATH to GCS Path on Kaggle
DATASET_PATH = 'Data'

MONET_FILENAMES = tf.io.gfile.glob(str(DATASET_PATH + '/monet_tfrec/*.tfrec'))
PHOTO_FILENAMES = tf.io.gfile.glob(str(DATASET_PATH + '/photo_tfrec/*.tfrec'))

monet_ds = load_dataset(MONET_FILENAMES, labeled=True).batch(1)
photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True).batch(1)

'''
example_monet = next(iter(monet_ds))
example_photo = next(iter(photo_ds))

plt.subplot(121)
plt.title('Photo')
plt.imshow(example_photo[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Monet')
plt.imshow(example_monet[0] * 0.5 + 0.5)

plt.show()
'''

with strategy.scope():
    monet_generator = Generator()  # transforms photos to Monet-esque paintings
    photo_generator = Generator()  # transforms Monet paintings to be more like photos

    # differentiates real Monet paintings and generated Monet paintings
    monet_discriminator = Discriminator()
    # differentiates real photos and generated photos
    photo_discriminator = Discriminator()
