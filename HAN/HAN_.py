from RISparser import readris
import pandas as pd
import warnings
import os
from collections import defaultdict
import re
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import recall_score,precision_score,accuracy_score
from torch.utils.data import DataLoader
import argparse
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
MAX_Sen_Len = 18
Max_SEN_Number = 16
MIN_SEN_Number = 3

def parseData(incl, excl):
    data = pd.DataFrame(columns=['content', 'label'])
    for l in incl:
        title = ''
        abstract = ''
        if 'abstract' in l:
            abstract = l['abstract']
        if 'primary_title' in l:
            title = l['primary_title']
        data = data.append({'content': title+". "+abstract, 'label': 1}, ignore_index=True) #从0开始索引

    for l in excl:
        title = ''
        abstract = ''
        if 'abstract' in l:
            abstract = l['abstract']
        if 'primary_title' in l:
            title = l['primary_title']
        data = data.append({'content': title+". "+abstract, 'label': 0}, ignore_index=True)

    data.title = data.content.fillna('none')
    data.reset_index(inplace=True,drop=True)  #重置索引
    return data

def judgeWords(str):
    """判断字符串是否包含字母以排除数字"""
    my_re = re.compile(r'[A-Za-z]', re.S)
    res = re.findall(my_re, str)
    if len(res):
        return True
    else:
        return False

def ConvertToid(dict,list):
    data = []
    for sen in list:
        indxs = []
        for word in sen:
            try:
                indx = dict[word]
            except KeyError:
                indx = 1
            indxs.append(indx)
        if len(indxs) < MAX_Sen_Len:
            indxs.extend([0]*(MAX_Sen_Len-len(indxs)))
        else:
            indxs = indxs[:MAX_Sen_Len]
        data.append(indxs)
    if len(data)<Max_SEN_Number:
        data.extend([[0]*MAX_Sen_Len]*(Max_SEN_Number-len(data)))
    return data

class covid_dataset(Dataset):
    def __init__(self, data,label):
        super().__init__()
        self.data = torch.tensor([item for item in data])
        self.label = torch.tensor([item for item in label])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

class HAN_Attention(nn.Module):
    '''层次注意力网络文档分类模型实现，词向量，句子向量'''
    def __init__(self, vocab_size, embedding_dim, gru_size, class_num, weights=None, is_pretrain=False):
        super(HAN_Attention, self).__init__()
        if is_pretrain:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)
        # 词注意力
        self.word_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True,dropout=0.25)
        self.word_query = nn.Parameter(torch.Tensor(2*gru_size, 1), requires_grad=True)   # 公式中的u(w)
        self.word_fc = nn.Linear(2*gru_size, 2*gru_size)
        # 句子注意力
        self.sentence_gru = nn.GRU(input_size=2*gru_size, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True,dropout=0.25)
        self.sentence_query = nn.Parameter(torch.Tensor(2*gru_size, 1), requires_grad=True)   # 公式中的u(s)
        self.sentence_fc = nn.Linear(2*gru_size, 2*gru_size)
        # 文档分类
        self.class_fc = nn.Linear(2*gru_size, class_num)

    def forward(self, x, use_gpu=True):  # x: b, sentence_num, sentence_len
        sentence_num = x.size(1)
        sentence_len = x.size(2)
        x = x.view(-1, sentence_len)  # b*sentence_num, sentence_len
        embed_x = self.word_embed(x)  # b*sentence_num , sentence_len, embedding_dim
        word_output, word_hidden = self.word_gru(embed_x)  # word_output: b*sentence_num, sentence_len, 2*gru_size
        # 计算u(it)
        word_attention = torch.tanh(self.word_fc(word_output))  # b*sentence_num, sentence_len, 2*gru_size
        # 计算词注意力向量weights: a(it)
        weights = torch.matmul(word_attention, self.word_query)  # b*sentence_num, sentence_len, 1
        weights = F.softmax(weights, dim=1)   # b*sentence_num, sentence_len, 1

        x = x.unsqueeze(2)  # b*sentence_num, sentence_len, 1
        if use_gpu:
            # 去掉x中padding为0位置的attention比重
            weights = torch.where(x!=0, weights, torch.full_like(x, 0, dtype=torch.float).cuda()) #b*sentence_num, sentence_len, 1
        else:
            weights = torch.where(x!=0, weights, torch.full_like(x, 0, dtype=torch.float))
        # 将x中padding后的结果进行归一化处理，为了避免padding处的weights为0无法训练，加上一个极小值1e-4
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)  # b*sentence_num, sentence_len, 1

        # 计算句子向量si = sum(a(it) * h(it)) ： b*sentence_num, 2*gru_size -> b*, sentence_num, 2*gru_size
        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sentence_num, word_output.size(2))

        sentence_output, sentence_hidden = self.sentence_gru(sentence_vector)  # sentence_output: b, sentence_num, 2*gru_size
        # 计算ui
        sentence_attention = torch.tanh(self.sentence_fc(sentence_output))  # sentence_output: b, sentence_num, 2*gru_size
        # 计算句子注意力向量sentence_weights: a(i)
        sentence_weights = torch.matmul(sentence_attention, self.sentence_query)   # sentence_output: b, sentence_num, 1
        sentence_weights = F.softmax(sentence_weights, dim=1)   # b, sentence_num, 1

        x = x.view(-1, sentence_num, x.size(1))   # b, sentence_num, sentence_len
        x = torch.sum(x, dim=2).unsqueeze(2)  # b, sentence_num, 1
        if use_gpu:
            sentence_weights = torch.where(x!=0, sentence_weights, torch.full_like(x, 0, dtype=torch.float).cuda())
        else:
            sentence_weights = torch.where(x!=0, sentence_weights, torch.full_like(x, 0, dtype=torch.float))  # b, sentence_num, 1
        sentence_weights = sentence_weights / (torch.sum(sentence_weights, dim=1).unsqueeze(1) + 1e-4)  # b, sentence_num, 1

        # 计算文档向量v
        document_vector = torch.sum(sentence_weights * sentence_output, dim=1)   # b, sentence_num, 2*gru_size
        document_class = self.class_fc(document_vector)   # b, sentence_num, class_num
        document_class = F.softmax(document_class, dim=1)
        return document_class

def train(model,loss_fn,optimizer,train_dataloader,epoch):
    model.train()
    timestep = 0
    print("*********第{}轮训练已经开始***********".format(epoch))
    for info in train_dataloader:
        text, targets = info
        texttensor = text.cuda()
        targets = targets.cuda()
        outputs = model(texttensor)
        loss = loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        timestep = timestep+args.batch_size
        if timestep % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, timestep,len(train_dataloader.dataset),
            100. *timestep / len(train_dataloader.dataset), loss.item()))
    print("********第{}轮训练完毕*********".format(epoch))

def test(model, loss_fn, test_loader):
    model.eval()
    loss = 0
    r = 0
    p = 0
    a = 0
    with torch.no_grad():
        for info in test_loader:
            text, targets = info
            texttensor = text.cuda()
            targets = targets.cuda()
            outputs = model(texttensor)
            loss = loss_fn(outputs, targets).item()  # sum up batch loss
            pred = outputs.argmax(dim=1).cpu()  # get the index of the max log-probability
            r = recall_score(targets.cpu(), pred)
            p = precision_score(targets.cpu(), pred)
            a = accuracy_score(targets.cpu(),pred)
            break
    print('\nTest set: Average loss: {:.4f}, recall: {}, precision: {}, accuracy: {} \n'.format(loss, r, p,a))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int, required=False,
                        help="Epoch to train.")
    parser.add_argument("--batch_size", default=16, type=int, required=False,
                        help="Batch size.")
    parser.add_argument("--lr", default=1e-4, type=float, required=False,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--embedding_dim", default=100, type=int, required=False,
                        help="the size of GRU hidden state")
    parser.add_argument("--weight_decay", default=1e-4, type=float, required=False,
                        help="Weight decay if we apply some.")
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test-batch-size', type=int, default=2000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                    help='Learning rate step gamma (default: 0.5)')
    args = parser.parse_args()


    warnings.filterwarnings('ignore')
    TrainingIncludes = readris(open("../DATA/1_Training_Included_20878.ris.txt", 'r', encoding="utf-8"))
    TrainingExcludes = readris(open('../DATA/2_Training_Excluded_38635.ris.txt', 'r', encoding="utf-8"))
    TrainingDf = parseData(TrainingIncludes, TrainingExcludes)
    stopwords = pd.read_csv('../DATA/pubmed.stoplist.csv')
    stopwords = stopwords.values.squeeze().tolist()

    Train_copy = TrainingDf.copy()
    Train_copy['Sen_Num'] = None
    word_freq = defaultdict(int)
    for i in range(0, len(Train_copy["content"])):
        Train_copy.at[i, "content"] = sent_tokenize(Train_copy.loc[i, "content"])
        Train_copy.loc[i, "Sen_Num"] = len(Train_copy.loc[i, "content"])
        s = []
        for sen in Train_copy.loc[i, "content"]:
            sen = re.sub(r'[.,"\'?:!;=><\\/]', ' ', sen)
            words = word_tokenize(sen)
            cutwords = [word.lower() for word in words if (word.lower() not in stopwords) and judgeWords(word)]
            for word in cutwords:
                word_freq[word] += 1
            s.append(cutwords)
        Train_copy.at[i, "content"] = s

    Train_copy = Train_copy.query('@MIN_SEN_Number<Sen_Num<@Max_SEN_Number')
    Train_copy.reset_index(inplace=True, drop=True)

    dict_filter = {voc: fre for voc, fre in word_freq.items() if fre >= 3}
    vocab = dict_filter.keys()
    int_vocab = dict(enumerate(vocab, 2))
    word_int = {w: int(i) for i, w in int_vocab.items()}
    int_vocab[0] = "<PAD>"
    int_vocab[1] = "<UNk>"
    word_int["<PAD>"] = 0
    word_int["<UNK>"] = 1


    np.save('int_vocab.npy', int_vocab)
    np.save('vocab_int.npy',word_int)
   
    Data_INT = Train_copy["content"].copy()
    for i in range(0, len(Data_INT)):
        Data_INT.at[i] = ConvertToid(word_int, Data_INT.iloc[i])

    x_train, x_val, y_train, y_val = train_test_split(Data_INT, Train_copy["label"], test_size=0.1, random_state=42)

    model = HAN_Attention(len(int_vocab.keys()), args.embedding_dim, 256, 2).cuda()
    # loss function
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    tran_set = covid_dataset(x_train, y_train)
    train_dataloader = DataLoader(tran_set, batch_size=args.batch_size, shuffle=True)
    Vali_set = covid_dataset(x_train, y_train)
    Vali_dataloader = DataLoader(Vali_set, batch_size=args.test_batch_size, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        train(model, loss_fn, optimizer, train_dataloader, epoch)
        test(model, loss_fn, Vali_dataloader)
        scheduler.step()
    # Save the model
    if args.save_model:
        torch.save(model.state_dict(), "HAN.pth")