import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyBertForSequenceClassification(BertPreTrainedModel):

    num_labels = 4
    num_tasks = 20

    def __init__(self, config):
        super(MyBertForSequenceClassification, self).__init__(config)
        self.num_labels = MyBertForSequenceClassification.num_labels
        self.num_tasks = MyBertForSequenceClassification.num_tasks

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 创建20个分类任务，每个任务共享输入： BertModel 的输出最后一层的 [CLS] 的 pooler_output
        # 但是源程序也说了，使用 [cls] 的 pooler_output is usually *not* a good summary
        # of the semantic content of the input, you're often better with averaging or pooling
        # the sequence of hidden-states for the whole input sequence.
        # module_list = []
        # for _ in range(self.num_tasks):
            # module_list.append(nn.Linear(config.hidden_size, self.num_labels))
        # self.classifier = nn.ModuleList(module_list)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels) for _ in range(self.num_tasks)])

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """forward

        :param input_ids:
        :param labels: 给定的形式是 [batch, num_tasks]
        """

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

#         logits = []
        # for i in range(self.num_tasks):
#             logits.append(self.classifier[i](pooled_output))

        logits = [self.classifier[i](pooled_output) for i in range(self.num_tasks)]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 这个要放在gpu 上，很容易遗忘，从而 loss.backward()的时候出错
            loss = torch.tensor([0.]).to(device)
            for i in range(self.num_tasks):
                loss += loss_fct(logits[i], labels[:,i])
            return loss 
        else:
            # 用于 验证集和测试集 标签的预测, 维度是[num_tasks, batch, num_labels]
            logits = [logit.cpu().numpy() for logit in logits]
            return torch.tensor(logits)

    # 可以选择 冻结 BertModel 中的参数，也可以不冻结，在 multiLabels classification 中不冻结,不调用该函数即可。这里给出了一个冻结的示范
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


