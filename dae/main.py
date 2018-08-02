import os
import shutil

import dae.autoencoder as autoencoder
from dae.utils.flags import FLAGS

_data_dir = FLAGS.data_dir
_chkpt_dir = FLAGS.chkpt_dir


def _check_and_clean_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.mkdir(d)


def main():
    if not os.path.exists(_data_dir):
        os.mkdir(_data_dir)

    _check_and_clean_dir(_chkpt_dir)

    os.mkdir(os.path.join(_chkpt_dir, "1"))
    os.mkdir(os.path.join(_chkpt_dir, "2"))
    os.mkdir(os.path.join(_chkpt_dir, "3"))
    os.mkdir(os.path.join(_chkpt_dir, "fine_tuning"))

    ae = autoencoder.main_unsupervised()
    autoencoder.main_supervised(ae)


if __name__ == "__main__":
    main()
