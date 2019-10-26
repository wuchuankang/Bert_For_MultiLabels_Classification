import os
import torch
import pandas as pd
from .config import args
from torch.utils.data import TensorDataset, DataLoader


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



def get_dataloader(tokenizer, data, batch_size, labels_available=True):
       
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


