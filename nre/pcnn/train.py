import warnings

from sklearn.metrics.base import UndefinedMetricWarning

from nre.pcnn.model.config import Config
from nre.pcnn.model.data_utils import getDataset
from nre.pcnn.model.pcnn_model import PCNNModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def main():
    # create instance of config
    config = Config()

    # build model
    model = PCNNModel(config)
    model.build()

    # create datasets
    dev = getDataset(config.filename_dev, config.processing_word, config.processing_relation, config.max_iter)
    train = getDataset(config.filename_train, config.processing_word, config.processing_relation, config.max_iter)

    # train model
    model.train(train, dev)


if __name__ == "__main__":
    main()
