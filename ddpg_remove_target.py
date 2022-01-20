import matplotlib.pyplot as plt
import numpy as np
import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 设置种子，使每次生成的随机数相同
np.random.seed(1)
tf.set_random_seed(1)

# 游戏
envirment = 'Pendulum-v0'

# 探索噪声
var = 3
# 重复训练
replicas=5
# 迭代的次数
episodes = 400
# 一次迭代进行的步数
steps = 200
# actor的学习率
lr_a = 0.001
# critic的学习率
lr_c = 0.001
# 未来奖励的衰减率
gamma = 0.9
# 更新target网络的策略，采用soft
replacement = [dict(name='soft', tau=0.01),dict(name='hard', ia=600, ic=500)][0]
# replay buffer的大小
memory_capacity = 10000
# 一次喂给网络的数据量
batch_size = 32
# 是否渲染画面
render = False

class Actor(object):
    def __init__(self, sess, action_dim, action_bound, lr, replacement):
        # tf.Session
        self.sess = sess
        # action的维度
        self.a_dim = action_dim
        # action取值的上界
        self.action_bound = action_bound
        # 学习率
        self.lr = lr
        # 更新target网络的策略
        self.replacement = replacement
        # 学习次数
        self.mem_counter = 0

        with tf.variable_scope('Actor'):
            # eval网络，可训练，s为输入
            self.a = self._build_net(S, scope='eval_net', trainable=True)
            # target网络，不训练，s_为输入
            # self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        # eval网络的变量
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        # target网络的变量
        # self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        # if self.replacement['name'] == 'hard':
        #     # hard策略：将target网络变量替换为eval网络变量
        #     self.mem_counter = 0
        #     self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        # else:
        #     # soft策略：逐步调整target网络变量向eval网络转变
        #     self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
        #                          for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # w的初始值的概率分布
            init_w = tf.random_normal_initializer(0., 0.3)
            # b的初始值为0.1
            init_b = tf.constant_initializer(0.1)
            # 全连接层，输入s，输出维度30，激活函数relu
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                # 全连接层，输入维度net，输出维度self.a_dim，激活函数tanh
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                # tanh的取值范围为(-1,1)，将结果乘以self.action_bound，scaled_a的取值范围为(-self.action_bound,self.action_bound)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')
        return scaled_a

    def learn(self, s):
        # 输入s，更新eval网络参数，最大化Q
        self.sess.run(self.train_op, feed_dict={S: s})
        # if self.replacement['name'] == 'soft':
        #     # 更新target网络参数
        #     self.sess.run(self.soft_replace)
        # else:
        #     # 每ia次学习，更新一次target网络参数
        #     if self.mem_counter % self.replacement['ia'] == 0:
        #         self.sess.run(self.hard_replace)
        #     self.mem_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]
        # eval网络self.a根据状态s选择采取的action
        return self.sess.run(self.a, feed_dict={S: s})[0]

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # 求self.a关于self.e_params的导数，即a对theta的导数，a_grads是相应的权重，即Q关于a的导数
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)
        with tf.variable_scope('A_train'):
            # 负学习率，最大化Q
            opt = tf.train.AdamOptimizer(-self.lr)
            # 将梯度self.policy_grads用于更新变量self.params，最大化Q
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, lr, gamma, replacement, a, a_):
        # tf.Session
        self.sess = sess
        # action的维度
        self.s_dim = state_dim
        # action取值的上界
        self.a_dim = action_dim
        # 学习率
        self.lr = lr
        # 未来奖励的衰减率
        self.gamma = gamma
        # 更新target网络的策略
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # 不计算a的梯度
            self.a = tf.stop_gradient(a)
            # eval网络，输入S，self.a，可训练
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)
            # target网络，输入S_，a_，不训练
            # self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)
            # eval网络的变量
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            # target网络的变量
            # self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            # target_q
            # self.target_q = R + self.gamma * self.q_
            self.target_q = R + self.gamma * self.q

        with tf.variable_scope('TD_error'):
            # target_q和self.q的差的平方的均值，也就是误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            # 最小化self.loss
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            # 对self.q求self.a的导数
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)

        # if self.replacement['name'] == 'hard':
        #     # hard策略：将target网络变量替换为eval网络变量
        #     self.mem_counter = 0
        #     self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        # else:
        #     # soft策略：逐步调整target网络变量向eval网络转变
        #     self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
        #                              for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            # w的初始值的概率分布
            init_w = tf.random_normal_initializer(0., 0.1)
            # b的初始值为0.1
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n1 = 30
                # w1_s大小[self.s_dim,n]，用init_w初始化
                w1_s = tf.get_variable('w1_s', [self.s_dim, n1], initializer=init_w, trainable=trainable)
                # w1_a大小[self.a_dim,n1]，用init_w初始化
                w1_a = tf.get_variable('w1_a', [self.a_dim, n1], initializer=init_w, trainable=trainable)
                # b1大小[1,n1]，用init_b初始化
                b1 = tf.get_variable('b1', [1, n1], initializer=init_b, trainable=trainable)
                # 第一层网络
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                # 第二层网络，[net,1]
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
        return q

    def learn(self, s, a, r, s_):
        # 输入s，更新eval网络参数，最大化Q
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        # if self.replacement['name'] == 'soft':
        #     # 更新target网络参数
        #     self.sess.run(self.soft_replacement)
        # else:
        #     # 每ic次学习，更新一次target网络参数
        #     if self.mem_counter % self.replacement['ic'] == 0:
        #         self.sess.run(self.hard_replacement)
        #     self.mem_counter += 1

# replay buffer
class Memory(object):
    def __init__(self, capacity, dims):
        # replay buffer的大小
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        # 数据量
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # 插入的位置
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # 随机选取n条数据
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

# 游戏
env = gym.make(envirment)
env = env.unwrapped
env.seed(1)
# 状态维数
state_dim = env.observation_space.shape[0]

# action维数
action_dim = env.action_space.shape[0]
# action取值范围上限
action_bound = env.action_space.high

# 占位符
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

sess = tf.Session()

actor = Actor(sess, action_dim, action_bound, lr_a, replacement)
# critic = Critic(sess, state_dim, action_dim, lr_c, gamma, replacement, actor.a, actor.a_)
critic = Critic(sess, state_dim, action_dim, lr_c, gamma, replacement, actor.a, actor.a)
# actor更新需要critic的梯度
actor.add_grad_to_graph(critic.a_grads)
# 初始化模型参数
sess.run(tf.global_variables_initializer())

M = Memory(memory_capacity, dims=2 * state_dim + action_dim + 1)

rewards=[]
for i in range(episodes):
    # 初始状态
    s = env.reset()
    # 记录steps步的奖励
    ep_reward = 0

    for j in range(steps):

        if render:
            # 渲染游戏
            env.render()

        # actor根据状态s选择a
        a = actor.choose_action(s)

        # 填加探索噪声
        a = np.clip(np.random.normal(a, var), -action_bound, action_bound)
        # 更新游戏状态
        s_, r, done, info = env.step(a)
        # 数据加入reply buffer
        M.store_transition(s, a, r, s_)

        if M.pointer > memory_capacity:
            # 衰减探索噪声
            var *= .9995  # decay the action randomness
            # 取batch_size个数据
            b_M = M.sample(batch_size)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]
            # 更新critic网络
            critic.learn(b_s, b_a, b_r, b_s_)
            # 更新actor网络
            actor.learn(b_s)

        s = s_
        ep_reward += r

        if j == steps - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            rewards.append(ep_reward)
            if ep_reward > -800:
                render = True

plt.plot(np.arange(0, episodes, 1), rewards, 'go--', color='blue', label='reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()