# -*- coding: utf-8 -*-
import torch
from config import args
from data_process import MultiLabelTextProcessor, get_dataloader
from model import MyBertForSequenceClassification
from transformers import BertTokenizer
from optimization import get_optimizer, get_scheduler
from training import train
from prediction import predict
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('args:{}'.format(args))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}".format(device, n_gpu))

logger.info('loading the model ...')

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = MyBertForSequenceClassification.from_pretrained('bert-base-chinese')
# 如果是调用保存的 参数模型，将模型参数路径放在和该模块同一路径，加入该模型的文件名为'save'，那么改为：
#model = MyBertForSequenceClassification.from_pretrained('./save')
# 迁移到 gpu 上
model.to(device)

logger.info('loaded the model')


logger.info('getting the data ...')
processor = MultiLabelTextProcessor('./my_data')
train_data = processor.get_data('sentiment_analysis_trainingset.csv')
eval_data = processor.get_data('sentiment_analysis_validationset.csv')
test_data = processor.get_data('sentiment_analysis_testset.csv', labels_available=False)
logger.info('already the data')

labels_list = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']
num_tasks = len(labels_list)
args['num_tasks'] = num_tasks
logger.info(" sentiment classes = %d", num_tasks)

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_data))
logger.info("  Batch size = %d", args['batch_size'])
logger.info("  Num steps = %d", int(len(train_data) / args['batch_size'] * args['num_train_epochs']))
 

logger.info('getting the dataloader ...')
train_dataloader = get_dataloader(tokenizer, train_data, args['batch_size'])
eval_dataloader = get_dataloader(tokenizer, eval_data, args['batch_size'])
test_dataloader = get_dataloader(tokenizer, test_data, args['batch_size'], labels_available=False)
logger.info('got the dataloader')


logger.info('get the optimizer')
optimizer = get_optimizer(model, lr=args['learning_rate'])

args['trian_total_steps'] = int(len(train_data) / args['batch_size'] * args['num_train_epochs'])

# warmup_steps 根据实际情况可以更改
logger.info('get the scheduler')
scheduler = get_scheduler(optimizer) 



logger.info('trainning ...')
train(args['num_train_epochs'], train_dataloader, eval_dataloader, model, optimizer, scheduler)


# Save a trained model
#已经在循环中保存了，不需要在保存一次了
#model.save_pretrained('./directory/to/save/')  

# re-load
#model = MyBertForSequenceClassification.from_pretrained('./directory/to/save/') 
# 迁移到 gpu 上
#model.to(device)

results = predict(model, test_data, test_dataloader, labels_list)
results.to_csv('results.csv', index=False)
