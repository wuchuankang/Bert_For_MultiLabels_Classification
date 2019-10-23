import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import BertTokenizer, MyBertForSequenceClassification
from transformers.optimization import AdamW, WarmupLinearSchedule
from sklearn.metrics import f1_score
from tqdm import tqdm_notebook as tqdm
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


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class MultiLabelTextProcessor():
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def get_data(self, filename, labels_available=True):
        data_df = pd.read_csv(os.path.join(self.data_dir, filename))      
        return self._create_data(data_df, labels_available)

    def _create_data(self, df,  labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(df.values):
            guid = row[0]
            text = row[1]
            if labels_available:
                labels = row[2:]
            else:
                labels = []
            examples.append(
                InputExample(guid=guid, text=text, labels=labels))
        return examples

processor = MultiLabelTextProcessor('./my_data')
train_data = processor.get_data('sentiment_analysis_trainingset.csv')
eval_data = processor.get_data('sentiment_analysis_validationset.csv')
test_data = processor.get_data('sentiment_analysis_testset.csv', labels_available=False)


labels_list = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']
num_tasks = len(labels_list)

def convert_examples_to_features(examples, max_seq_length, tokenizer, labels_available=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)


        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if labels_available:
            labels_ids = []
            for label in example.labels:
                labels_ids.append(label)    

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_ids=labels_ids))
        else:
            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,))
    return features


def get_dataloader(data, batch_size, labels_available=True):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    features = convert_examples_to_features(data, args['max_seq_length'], tokenizer, labels_available)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
   
    if labels_available:
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)  
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader

train_dataloader = get_dataloader(train_data, args['batch_size'])
eval_dataloader = get_dataloader(eval_data, args['batch_size'])
test_dataloader = get_dataloader(test_data, args['batch_size'], labels_available=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}".format(device, n_gpu))


model = MyBertForSequenceClassification.from_pretrained('bert-base-chinese')
# 迁移到 gpu 上
model.to(device)


def get_optimizer(model, lr):       

    # Prepare optimiser and schedule 
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    return AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)


optimizer = get_optimizer(model, lr=args['learning_rate'])

num_epochs = args['num_train_epochs']
trian_total_steps = int(len(train_data) / args['batch_size'] * num_epochs)


# warmup_steps 根据实际情况可以更改
warmup_steps = args['warmup_steps'] 
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=trian_total_steps)


def train(num_epocs):

    model.train()
    for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):

        train_loss = 0
        num_train, train_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # 将运算数据迁移到 gpu 上
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)

            loss.backward()

            train_loss += loss.item()
            num_train += input_ids.size(0)
            train_steps += 1
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        logger.info('Train loss after epoc {}'.format(train_loss / train_steps))
        logger.info('Eval after epoc {}'.format(i_+1))
        
        eval()

def eval():
    
    all_logits = None
    all_labels = None
    
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    eval_steps, num_eval = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        # 将运算数据迁移到 gpu 上
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

            
            
        # 将各个batch 的 logits 和 labels 拼接在一起，用于 f1_score 计算
        # logits 维度 [num_tasks, batch, 4], label_ids ： [ batch, num_tasks]
        # all_logits :[num_tasks, num_eval, 4], all_label_ids : [num_eval, num_tasks]
        
        if all_logits is None:
            all_logits = logits.detach().cpu()
        else:
            all_logits = torch.cat((all_logits, logits.detach().cpu()), axis=1)
            
        if all_labels is None:
            all_labels = label_ids.detach().cpu()
        else:    
            all_labels = torch.cat((all_labels, label_ids.detach().cpu()), axis=0)
        
        # 可以在这里添加一个 assert 判断！
        
        eval_loss += tmp_eval_loss.mean().item()

        num_eval += input_ids.size(0)
        eval_steps += 1

    eval_loss = eval_loss / eval_steps
    
    # Compute f1_scores
    f1_scores_list = []
    # pred_labels : [num_tasks, num_eval]
    pred_labels = torch.argmax(all_logits, dim=2)
    for i in range(num_tasks):
        f1_scores_list.append(f1_score(all_labels[:,i].numpy(), pred_labels[i].numpy(), average='macro'))
        
    f1_score  = np.mean(f1_scores_list)
    
    logger.info('Eval loss after epoc {}'.format(eval_loss))
    logger.info('f1_score after epoc {}'.format(f1_score))


train(args['num_train_epochs'])


# Save a trained model
model.save_pretrained('./directory/to/save/')  

# re-load
#model = MyBertForSequenceClassification.from_pretrained('./directory/to/save/') 
# 迁移到 gpu 上
#model.to(device)

def predict():
    
    # Hold input data for returning it 
    input_data = [{ 'id': input_test.guid, 'content': input_test.text } for input_test in test_data]
    
    all_logits = None
    model.eval()
    for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
        input_ids, input_mask, segment_ids = batch
        # 将运算数据迁移到 gpu 上
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        if all_logits is None:
            all_logits = logits.detach().cpu()
        else:
            all_logits = torch.cat((all_logits, logits.detach().cpu()), axis=1)
        
    # pred_labels : [num_tasks, num_eval]
    pred_labels = torch.argmax(all_logits, dim=2) 
    
    # 因为预处理将标签 +2，所以这里再减去2
    pred_labels -= 2
    return pd.merge(pd.DataFrame(input_data), pd.DataFrame(pred_labels.numpy(), columns=labels_list), left_index=True, right_index=True)

results = predict()
pd.to_csv(results, index=False)
