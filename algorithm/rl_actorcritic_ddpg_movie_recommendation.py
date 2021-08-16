import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import random
import matplotlib.pyplot as plt
from torch.optim import Adam
from collections import defaultdict
from collections import Counter

# 加载数据
ratings_list = [i.strip().split("::") for i in open(r'ml-1m\ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open(r'ml-1m\users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open(r'ml-1m\movies.dat', encoding = 'latin-1').readlines()]
# 格式化数据为DataFrame二维表
ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

# 评分矩阵
R_df = ratings_df.pivot(index = 'UserID', columns = 'MovieID', values = 'Rating').fillna(0)
R_df = R_df.astype(int)

# 所有用户id的list
userids = list(R_df.index.values)
# 所有用户id的dictionary, {index: userid}
idx_to_userids = {i: userids[i] for i in range(len(userids))}
# 所有用户id的dictionary, {userid: index}
userids_to_idx = {userids[i]: i for i in range(len(userids))}

# 所有电影id的list
columns = list(R_df)
idx_to_id = {i: columns[i] for i in range(len(columns))}
id_to_idx = {columns[i]: i for i in range(len(columns))}

"""##Getting Embeddings of User and Item(Movie Id's)"""

# R评分矩阵列表形式
R = R_df.values
# 每个用户评分平均值
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# U 用户向量
# Vt 电影向量的转置
U, sigma, Vt = svds(R_demeaned, k = 100)

# 电影向量
V = Vt.transpose()

# 电影向量的列表; len(movie_list) = 3706
movie_list = V.tolist()
# {movieid: 电影向量} len(movie_embeddings_dict) = 3076
movie_embeddings_dict = {columns[i]: np.array(movie_list[i], dtype = np.float32) for i in range(len(columns))}

# 用户向量列表; len(user_list = 6040)
user_list = U.tolist()
# {userid: 用户向量}
user_embeddings_dict = {userids[i]: np.array(user_list[i], dtype = np.float32) for i in range(len(userids))}

# 所有评分
users_df = ratings_df.sort_values(["UserID", "Timestamp"]).set_index("UserID").fillna(0).drop("Timestamp", axis = 1)
# 所有评分记录按照用户分组
users = dict(tuple(users_df.groupby("UserID")))

"""##Train and Test Dataset"""

# userid, 打4或5分总数超过10的用户评价过的电影id以及评分
users_dict = defaultdict(dict)
# 打4或5分总数超过10的用户id(5950)
users_id_list = set()

# 构造user_dict user_id_list
for user_id in users:
    rating_freq = Counter(users[user_id]["Rating"].values)
    if rating_freq['4'] + rating_freq['5'] < 10:
        continue
    else:
        users_id_list.add(user_id)
        users_dict[user_id]["item"] = users[user_id]["MovieID"].values
        users_dict[user_id]["rating"] = users[user_id]["Rating"].values

# 打4或5分总数超过10的用户id(5950)
users_id_list = np.array(list(users_id_list))

# 划分训练集75%，测试集25%
train_users, test_users = train_test_split(users_id_list)


class UserDataset(Dataset):
    def __init__(self, users_list, users_dict):
        self.users_list = users_list
        self.users_dict = users_dict

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, idx):
        user_id = self.users_list[idx]
        items = [('1',)] * 10
        ratings = [('0',)] * 10
        j = 0
        for i, rate in enumerate(self.users_dict[user_id]["rating"]):
            if int(rate) > 3 and j < 10:
                items[j] = self.users_dict[user_id]["item"][i]
                ratings[j] = self.users_dict[user_id]["rating"][i]
                j += 1
        # item = list(self.users_dict[user_id]["item"][:])
        # rating = list(self.users_dict[user_id]["rating"][:])
        size = len(items)

        return {'item': items, 'rating': ratings, 'size': size, 'userid': user_id, 'idx': idx}


train_users_dataset = UserDataset(train_users, users_dict)
test_users_dataset = UserDataset(test_users, users_dict)

train_dataloader = DataLoader(train_users_dataset, batch_size = 1)
test_dataloader = DataLoader(test_users_dataset, batch_size = 1)

train_num = len(train_dataloader)

"""#State Representation Models"""


# 状态的表达方式
def drrave_state_rep(userid_b, items, memory, idx):
    user_num = idx
    H = []  # item embeddings
    user_n_items = items
    user_embeddings = torch.Tensor(np.array(user_embeddings_dict[userid_b[0]]), ).unsqueeze(0)
    for i, item in enumerate(user_n_items):
        H.append(np.array(movie_embeddings_dict[item[0]]))
    avg_layer = nn.AvgPool1d(1)
    item_embeddings = avg_layer(torch.Tensor(H, ).unsqueeze(0)).permute(0, 2, 1).squeeze(0)
    state = torch.cat([user_embeddings, user_embeddings * item_embeddings.t(), item_embeddings.t()])
    return state  # state tensor shape [21,100]


"""#Actor, Critic Module"""


# Actor Model:
# Generating an action a based on state s

class Actor(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Actor, self).__init__()

        # 0.5 是指该层（layer）的神经元在每次迭代训练时会随机有 50% 的可能性被丢弃（失活），不参与训练，防止过拟合
        self.drop_layer = nn.Dropout(p = 0.5)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        # state = self.state_rep(state)
        x = F.relu(self.linear1(state))
        # print(x.shape)
        x = self.drop_layer(x)
        x = F.relu(self.linear2(x))
        # print(x.shape)
        x = self.drop_layer(x)
        # x = torch.tanh(self.linear3(x)) # in case embeds are -1 1 normalized
        x = self.linear3(x)  # in case embeds are standard scaled / wiped using PCA whitening
        # return state, x
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Critic, self).__init__()

        self.drop_layer = nn.Dropout(p = 0.5)

        self.linear1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        # print(x.shape)
        x = F.relu(self.linear1(x))
        x = self.drop_layer(x)
        x = F.relu(self.linear2(x))
        x = self.drop_layer(x)
        x = self.linear3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        # 容量
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    # 将(S, A, R, S')的四元组放入缓冲池
    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    # 从缓冲池中取出四元组
    def sample(self, batch_size):
        # 从buffer中随即取出batch_size个元素，也就是1个四元组
        batch = random.sample(self.buffer, batch_size)
        # print(batch)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


device = 'cpu'

p_loss = []
v_loss = []


# 更新四个网络
def ddpg_update(batch_size = 1, gamma = 0.6, min_value = -np.inf, max_value = np.inf, soft_tau = 1e-2):
    state, action, reward, next_state = replay_buffer.sample(batch_size)
    state = torch.FloatTensor(state).to(device)

    next_state = torch.FloatTensor(next_state).to(device)

    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    # print(state.shape)
    # print(policy_net(state).shape)
    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()
    p_loss.append(policy_loss)
    next_action = target_policy_net(next_state)
    target_value = target_value_net(next_state, next_action.detach())
    expected_value = reward + gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    # print("1")
    value_loss = value_criterion(value, expected_value.detach())
    # print("2")
    v_loss.append(value_loss)
    policy_optimizer.zero_grad()
    # print("3")
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


# 定义网络
# 参数input_dim，output_dim，hidden_dim
value_net = Critic(2100, 100, 256)
policy_net = Actor(2100, 100, 256)
target_value_net = Critic(2100, 100, 256)
target_policy_net = Actor(2100, 100, 256)

# ?
target_policy_net.eval()
target_value_net.eval()
# 初始化target网络
for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

# 损失函数
value_criterion = nn.MSELoss()
# 优化器
value_optimizer = Adam(value_net.parameters(), lr = 1e-4)
policy_optimizer = Adam(policy_net.parameters(), lr = 1e-4)

# 初始化缓冲池
replay_buffer_size = 10000
replay_buffer = ReplayBuffer(replay_buffer_size)

# 4462*10的-1矩阵
memory = np.ones((train_num, 10)) * -1


def get_action(action_emb, userid_b, preds):
    # 用户的偏好向量
    action_emb = torch.reshape(action_emb, [1, 100]).unsqueeze(0)
    # 用户评价过的所有电影的对应的向量列表
    item_embedding = []
    # userid_b[0]用户评价过的所有电影中
    for movie in users_dict[userid_b[0]]["item"]:
        # movie_embeddings_dict[movie]为id为movie的电影对应的向量
        item_embedding.append(np.array(movie_embeddings_dict[movie]))
    # 将用户评价过的电影的向量转换成张量
    item_embedding = torch.Tensor(item_embedding, )
    # .t()张量转置
    # .unsqueeze(0)增加维度
    items = item_embedding.t().unsqueeze(0)
    # .squeeze(0)降低维度
    # 用户偏好的电影向量与所有电影向量点乘
    m = torch.bmm(action_emb, items).squeeze(0)
    # 从大到小排序，选出相似度最高的电影
    sorted_m, indices = torch.sort(m, descending = True)
    index_list = list(indices[0])
    print('index_list', index_list)
    for i in index_list:
        if users_dict[userid_b[0]]["item"][i] not in preds:
            preds.add(users_dict[userid_b[0]]["item"][i])
            return int(i)


def update_memory(memory, action, idx):
    memory[idx] = list(memory[idx, 1:]) + [action]


rate = 0

"""#Training"""

# dict()创建新字典
preddict = dict()
# iter()迭代器
it = iter(train_dataloader)
# tqdm进度条
for episode in tqdm.tqdm(range(train_num - 1)):
    batch_size = 1
    # set()创建无序不重复集合
    preds = set()
    # 某用户的十条评分记录
    first = next(it)
    # item_b：10个电影id。rating_b：10条评分。size_b：。userid_b：用户id。idx_b：
    item_b, rating_b, size_b, userid_b, idx_b = first['item'], first['rating'], first['size'], first['userid'], first[
        'idx']
    # 把10个电影id存到memory
    memory[idx_b] = [item[0] for item in item_b]
    # 当前状态 参数：用户id，10部电影id，
    state = drrave_state_rep(userid_b, item_b, memory, idx_b)
    for j in range(5):
        state_rep = torch.reshape(state, [-1])
        # 输出用户偏好的电影向量，用于在电影中
        action_emb = policy_net(state_rep)
        # 推荐的电影的序号
        action = get_action(action_emb, userid_b, preds)
        # 实际用户对推荐的这部电影给出的评分
        rate = int(users_dict[userid_b[0]]["rating"][action])
        try:
            ratings = (int(rate) - 3) / 2
        except:
            ratings = 0
        reward = torch.Tensor((ratings,))

        if reward > 0:
            update_memory(memory, int(users_dict[userid_b[0]]["item"][action]), idx_b)

        next_state = drrave_state_rep(userid_b, item_b, memory, idx_b)
        next_state_rep = torch.reshape(next_state, [-1])
        replay_buffer.push(state_rep.detach().cpu().numpy(), action_emb.detach().cpu().numpy(), reward,
                           next_state_rep.detach().cpu().numpy())
        if len(replay_buffer) > batch_size:
            ddpg_update()

        state = next_state
    preddict[userid_b[0]] = preds

plt.plot(v_loss)

plt.plot(p_loss)

"""#Testing"""

# prediction algorithm
it2 = iter(test_dataloader)
# 初始化准确率
precision = 0
# 给出的预测字典
test_pred_dict = dict()
for j in range(len(test_dataloader) - 1):
    # 拿到迭代对象
    first = next(it2)
    item_b, rating_b, size_b, userid_b, idx_b = first['item'], first['rating'], first['size'], first['userid'], first[
        'idx']
    memory[idx_b] = [item[0] for item in item_b]
    state = drrave_state_rep(userid_b, item_b, memory, idx_b)
    count = 0
    test_pred = set()
    for j in range(5):
        state_rep = torch.reshape(state, [-1])
        action_emb = policy_net(state_rep)
        action = get_action(action_emb, userid_b, test_pred)
        rate = int(users_dict[userid_b[0]]["rating"][action])
        try:
            rating = (int(rate) - 3) / 2
        except:
            rating = 0
        reward = torch.Tensor((rating,))

        if reward > 0:
            count += 1
            update_memory(memory, int(users_dict[userid_b[0]]["item"][action]), idx_b)
        next_state = drrave_state_rep(userid_b, item_b, memory, idx_b)
        state = next_state
    #
    precision += count / 5
    test_pred_dict[userid_b[0]] = test_pred
np.save('test_pred_dict_small_5', test_pred_dict)
print("p", precision / (len(test_dataloader) - 1))
