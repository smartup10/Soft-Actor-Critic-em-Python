import numpy as np
import random
import gym
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import Adam

# Função para converter um número inteiro em one-hot
def to_one_hot(x, n):
    ret = np.zeros(n)
    ret[x] = 1
    return ret

# Função para calcular a recompensa acumulada ao longo do tempo
def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros(n)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

# Classe que representa a política do agente
class Policy:
    def __init__(self, num_inputs, num_actions):
        self.w1 = Linear(num_inputs, 32)
        self.w2 = Linear(32, num_actions)

    def __call__(self, x):
        return x.sequential([self.w1, Tensor.tanh, self.w2])

    def sample(self, obs):
        act_distribution = self(obs).softmax()
        act = [np.random.choice(2, p=act_distribution.numpy()[i]) for i in range(act_distribution.shape[0])]
        return act, act_distribution

# Classe que representa a função Q do agente
class QNetwork:
    def __init__(self, num_inputs, num_actions):
        self.w1 = Linear(num_inputs + num_actions, 32)
        self.w2 = Linear(32, 32)
        self.w3 = Linear(32, 1)

    def __call__(self, x):
        return x.sequential([self.w1, Tensor.relu, self.w2, Tensor.relu, self.w3])

# Função para aprender a função Q
def learn_q_function(samples):
    sa = Tensor([s + to_one_hot(a, 2) for s, a, _, _, _ in samples])
    r = [x[2] for x in samples]
    mask = [int(not x[3]) for x in samples]
    nsa = Tensor([x[4] + to_one_hot(policy.sample(x[4])[0], 2) for x in samples])

    next_value = Tensor(r) + GAMMA * np.min([q1(nsa).numpy(), q2(nsa).numpy()], axis=0) * Tensor(mask)
    next_value = next_value.reshape(-1, 1)

    q1_t, q2_t = q1(sa), q2(sa)
    q_loss = ((q1_t - next_value) ** 2).mean() + ((q2_t - next_value) ** 2).mean()
    
    q_opt.zero_grad()
    q_loss.backward()
    q_opt.step()

    for x, y in zip(hard_params, soft_params):
        y.assign((1 - TAU) * y + TAU * x)

    return q_loss.numpy()[0]

# Função para executar um episódio
def run_episode():
    done = False
    state, done = env.reset(), False
    rews = []

    while not done:
        act_distribution = policy(Tensor(state)).softmax().numpy()
        ent = -sum(act_distribution * np.log2(act_distribution))
        act = np.random.choice(2, p=act_distribution)
        next_state, rew, done, _ = env.step(act)
        
        if done and episode_steps < env._max_episode_steps:
            rew = -10.0
        rews.append(rew + ALPHA * ent)
        
        memory.append((state, act, rew, done, next_state))
        state = next_state

    return rews

# Parâmetros globais
BS = 256
ALPHA = 0.2
GAMMA = 0.99
TAU = 0.005

# Inicialização do ambiente CartPole
env = gym.make("CartPole-v1")
num_inputs, num_actions = env.observation_space.shape[0], env.action_space.n

# Inicialização da política e da função Q
policy = Policy(num_inputs, num_actions)
q1 = QNetwork(num_inputs, num_actions)
q2 = QNetwork(num_inputs, num_actions)
q1s = QNetwork(num_inputs, num_actions)
q2s = QNetwork(num_inputs, num_actions)

hard_params = q1.parameters() + q2.parameters()
soft_params = q1s.parameters() + q2s.parameters()

for x, y in zip(hard_params, soft_params):
    y.assign(x + 0)

# Inicialização dos otimizadores
q_opt = Adam(q1.parameters() + q2.parameters(), lr=1e-2)
policy_opt = Adam(policy.parameters(), lr=1e-2)

# Lista para armazenar as recompensas e perdas ao longo do treinamento
env_steps = []
env_losses = []

# Treinamento principal
for episode in range(300):
    # Executa um episódio e coleta as recompensas
    rews = run_episode()

    # Aprendizado da política
    X = [x[0] for x in memory]
    Y = [x[1] for x in memory]
    tmp = np.zeros((len(Y), 2))
    tmp[range(len(Y)), Y] = reward_to_go(rews)

    policy_loss = -(policy(Tensor(X)).logsoftmax() * Tensor(tmp)).mean()
    
    policy_opt.zero_grad()
    policy_loss.backward()
    policy_opt.step()

    # Atualização dos parâmetros da política e da função Q
    if len(memory) >= BS:
        q_loss = learn_q_function([memory[random.choice(len(memory))] for _ in range(BS)])
    else:
        q_loss = None

    env_steps.append(len(rews))
    env_losses.append((q_loss, policy_loss))

# Plota resultados
import matplotlib.pyplot as plt

plt.plot(env_steps)
plt.show()

plt.plot([np.sum(rews) for rews in memory])
plt.show()
