import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = {
    "max_seq_length": 512,
    "batch_size": 32,
    "learning_rate": 3e-5,
    "num_train_epochs": 4,
    "warmup_steps": 20000
}

logger.info('args:{}'.format(args))
logger.info('get the data ...')

