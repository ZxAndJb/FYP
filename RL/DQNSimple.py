import collections
import random
import torch.nn.functional as F
from tqdm import tqdm
from EnvSimple import Env
from RISparser import readris
import pandas as pd
import warnings
from collections import defaultdict
import re
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.model_selection import train_test_split
from HAN_ import parseData,judgeWords,ConvertToid,covid_dataset,HAN_Attention
import torch
import numpy as np
import matplotlib.pyplot as plt


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

HookInput = None

def getArticleVector(module, input, output):
    global HookInput
    HookInput = input

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        state = np.array(state)
        next_state = np.array(next_state)
        return state, action, reward, next_state, done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

class DQN:
    ''' DQN算法,包括Double DQN '''
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 weight_decay,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='DoubleDQN'):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim,  hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim,  hidden_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), weight_decay=weight_decay,
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state, mode):
        if mode == 'Train':
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                action= self.q_net(state).argmax().item()
        else:
            action = self.q_net(state).argmax().item()
        return action


    def update(self, transition_dict):
        states = torch.stack(list(transition_dict['states'])).squeeze().to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.stack(list(transition_dict['next_states'])).squeeze().to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

def envConvert(state,model):
    state = state.unsqueeze(0)
    with torch.no_grad():
        model(state)
    return HookInput[0]

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ## 加载字典
    int_vocab = np.load('../HAN/int_vocab.npy',allow_pickle=True).item()
    vocab_int = np.load('../HAN/vocab_int.npy',allow_pickle=True).item()
    TrainingIncludes = readris(open("../DATA/1_Training_Included_20878.ris.txt", 'r', encoding="utf-8"))
    TrainingExcludes = readris(open('../DATA/2_Training_Excluded_38635.ris.txt', 'r', encoding="utf-8"))
    TrainingDf = parseData(TrainingIncludes, TrainingExcludes)
    stopwords = pd.read_csv('../DATA/pubmed.stoplist.csv')
    stopwords = stopwords.values.squeeze().tolist()
    Train_copy = TrainingDf.copy()
    Train_copy['Sen_Num'] = None
    MAX_Sen_Len = 18
    Max_SEN_Number = 16
    MIN_SEN_Number = 3

    for i in range(0, len(Train_copy["content"])):
        Train_copy.at[i, "content"] = sent_tokenize(Train_copy.loc[i, "content"])
        Train_copy.loc[i, "Sen_Num"] = len(Train_copy.loc[i, "content"])
        s = []
        for sen in Train_copy.loc[i, "content"]:
            sen = re.sub(r'[.,"\'?:!;=><\\/]', ' ', sen)
            words = word_tokenize(sen)
            cutwords = [word.lower() for word in words if (word.lower() not in stopwords) and judgeWords(word)]
            s.append(cutwords)
        Train_copy.at[i, "content"] = s

    Train_copy = Train_copy.query('@MIN_SEN_Number<Sen_Num<@Max_SEN_Number')
    Train_copy.reset_index(inplace=True, drop=True)


    Data_INT = Train_copy["content"].copy()
    for i in range(0, len(Data_INT)):
        Data_INT.at[i] = ConvertToid(vocab_int, Data_INT.iloc[i])

    x_train, x_val, y_train, y_val = train_test_split(Data_INT, Train_copy["label"], test_size=0.1, random_state=42)

    # loss function
    tran_set = covid_dataset(x_train, y_train)
    Vali_set = covid_dataset(x_train, y_train)

    model = HAN_Attention(len(int_vocab.keys()),100,256,2).cuda()
    model.load_state_dict(torch.load("../HAN/HAN.pth"))
    model.class_fc.register_forward_hook(getArticleVector)
    Pre_train_embedding = model.word_embed.weight

    lr = 1e-4
    num_episodes = 2000
    weight_decay = 1e-4
    hidden_dim = 256
    gamma = 0.98
    epsilon = 0.4
    target_update = 15
    buffer_size = 30000
    minimal_size = 1000
    batch_size = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = 512
    action_dim = MAX_Sen_Len * Max_SEN_Number
    Env = Env(np.array(x_train[0]).shape, MAX_Sen_Len * Max_SEN_Number, 200, model, tran_set)
    agent = DQN(state_dim, hidden_dim, action_dim, lr,weight_decay, gamma, epsilon,
                target_update, device,'Qnet')

    return_list = []
    for i in range(20):
        with tqdm(total=int(num_episodes / 20), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 20)):
                episode_return = 0
                state = Env.reset().to(device)
                state = envConvert(state,model)
                done = False
                while not done:
                    action = agent.take_action(state,'Train')
                    next_state, reward, done = Env.step(action)
                    next_state = envConvert(next_state,model)
                    episode_return += reward
                    replay_buffer.add(state.cpu(), action, reward, next_state.cpu(), done)
                    state = next_state

                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 20 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    agent.savemodel()
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format('Training Dataset'))
    plt.show()

