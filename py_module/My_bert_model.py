
class MyBertForSequenceClassification(BertPreTrainedModel):

    """
    # 有6大类，下面共有20个标签，不考虑大类，直接考虑标签，每个标签有4个类别，可以将
    # 任务分为多任务分类问题，每一个任务都是一个4分类问题。
    # 参见 https://github.com/brightmart/sentiment_analysis_fine_grain
    """


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

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        """forward

        :param input_ids:
        :param labels: 给定的形式是 [batch, num_tasks]
        """

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

#         logits = []
        # for i in range(self.num_tasks):
#             logits.append(self.classifier[i](pooled_output))

        logits = [self.classifier[i](pooled_output) for i in range(self.num_tasks)]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = torch.tensor([0.])
            for i in range(self.num_tasks):
                loss += loss_fct(logits[i], labels[:,i])
            return loss 
        else:
            # 用于 验证集和测试集 标签的预测, 维度是[num_tasks, batch, num_labels]
            return torch.tensor(logits)


    # 可以选择 冻结 BertModel 中的参数，也可以不冻结，在 multiLabels classification 中不冻结,不调用该函数即可。这里给出了一个冻结的示范
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

