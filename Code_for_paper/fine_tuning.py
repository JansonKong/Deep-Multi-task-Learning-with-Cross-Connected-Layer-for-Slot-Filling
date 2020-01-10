from __future__ import print_function
import os
import logging
import sys
from .networks.BiLSTM import BiLSTM
from .util.preprocessing import perpare_dataset
from .util.preprocessing import load_dataset_pickle


def train():
    """
    Multi-task train for three task:
    1. Computer slot task
    2. Phone slot task
    3. Camera slot task

    Model:
    BiLSTM-CRF with the share BiLSTM layer and share embedding(50)

    :return:
    """
    # :: Change into the working dir of the script ::
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # :: Logging level ::
    logging_level = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging_level)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    datasets = {
        'camera':
            {'columns': {0: 'tokens', 1: 'camera_BIO'},
             'label': 'camera_BIO',
             'evaluate': True,
             'comment_symbol': None},
        'computer':
            {'columns': {0: 'tokens', 1: 'computer_BIO'},  # 0 for the feature, 1 for the label
             'label': 'computer_BIO',
             'evaluate': True,
             'comment_symbol': None},
        'phone':
            {'columns': {0: 'tokens', 1: 'phone_BIO'},
             'label': 'phone_BIO',
             'evaluate': True,
             'comment_symbol': None}
    }

    embeddings_path = 'wiki_100.utf8'

    # :: Prepares the data set to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
    pickle_file = perpare_dataset(embeddings_path, datasets)

    # Load the embeddings and the data set
    embeddings, mappings, data = load_dataset_pickle(pickle_file)

    phone_model_path = 'models/phone_0.9080_0.8964_28.h5'
    # :: set the model ::
    phone_model = BiLSTM.load_model(phone_model_path)
    phone_model.set_dataset(datasets, data)
    phone_model.model_save_path = "models/[model_name]_[DevScore]_[TestScore]_[Epoch].h5"
    phone_model.sentence_result_path = 'sentence_result/'

    # :: train ::
    phone_model.fine_tuning(epochs=100, model_name='phone')


if __name__ == '__main__':
    # :: train and save the best model ::
    train()
