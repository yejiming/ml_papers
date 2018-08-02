import numpy as np

from mann.utils import OmniglotDataLoader, one_hot_decode, five_hot_decode


data_loader = OmniglotDataLoader(
    image_size=(20, 20),
    n_train_classses=1200,
    n_test_classes=423
)

x_image, x_label, y = data_loader.fetch_batch(5, 16, 50, type="train", augment=True, label_type="one_hot")
print(np.array(x_image).shape, np.array(x_label).shape, np.array(y).shape)
