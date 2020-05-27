import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow.keras import Model
import cirq
from cirq.contrib.svg import SVGCircuit
import sympy
import numpy as np
import gym
import random
from collections import namedtuple


Transition = namedtuple('Transition',('state', 'action', 'reward', 'next_state', 'done'))

nq = 4
nl = 2
q = cirq.GridQubit.rect(1, nq) # Define qubit grid. In this case

symb = sympy.symbols('theta0:'+str(4+3*nq*nl))
W = np.array(symb[4:]).reshape(nl, nq, 3)
circuit = cirq.Circuit()
for i in range(4):
        circuit+=cirq.rx(symb[i])(q[i])
        circuit+=cirq.rz(symb[i])(q[i])
for l in range(nl):
    for i in range(3):
        circuit.append(cirq.CNOT(q[i], q[i + 1]))
    for i in range(4):
        print(W[l][i][0])
        circuit+=cirq.rz(W[l][i][0])(q[i])
        circuit+=cirq.ry(W[l][i][1])(q[i])
        circuit+=cirq.rz(W[l][i][2])(q[i])

op = cirq.Z(q[0]), cirq.Z(q[1]), cirq.Z(q[2]), cirq.Z(q[3])


print(SVGCircuit(circuit))

class VQC(Model):
    def __init__(self):
        super(VQC, self).__init__()
        self.step=0
        self.TARGET_UPDATE = 20
        self.optimizer = tf.keras.optimizers.SGD(lr=0.5)
        self.W = tf.Variable(np.random.rand(3*nq*nl))
        self.bias=tf.Variable(np.zeros(4),dtype='float32')
        self.oW = tf.Variable(self.W.initialized_value(), trainable=0)
        self.obias = tf.Variable(self.bias.initialized_value(),dtype='float32', trainable=0)


    @tf.function
    def __call__(self,inputs,now):
        if now:
            return tfq.layers.Expectation()(circuit, symbol_names=symb, symbol_values=[tf.concat([inputs, self.W], axis=0)],operators=op)[0] + self.bias

        else:
            return tfq.layers.Expectation()(circuit, symbol_names=symb, symbol_values=[tf.concat([inputs, self.oW], axis=0)],operators=op)[0] + self.obias

    def train_step(self,batch):
        self.step += 1
        with tf.GradientTape() as tape:
            t=[encode(4, item.next_state)for item in batch]
            labels = [batch[i].reward + (1 - int(batch[i].done)) * gamma * tf.reduce_max(self(t[i],0)) for i in range(len(batch))]
            predictions = [self(encode(4, item.state),1)[item.action] for item in batch]
            loss = sum((labels[i]-predictions[i])**2 for i in range(4)) / len(batch)
            print('loss:',loss)
        gradients = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        if self.step%self.TARGET_UPDATE==0:
            self.oW = tf.Variable(self.W.initialized_value(), trainable=0)
            self.obias = tf.Variable(self.bias.initialized_value(), trainable=0)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def output_all(self):
        return self.memory

    def __len__(self):
        return len(self.memory)


def encode(_length, _decimal):
    binNum = bin(int(_decimal))[2:]
    outputNum = [int(item) for item in binNum]
    if len(outputNum) < _length:
        outputNum = np.concatenate((np.zeros((_length - len(outputNum),)), np.array(outputNum)))
    else:
        outputNum = np.array(outputNum)
    outputNum=tf.Variable(outputNum,trainable=0,dtype='float64')
    return outputNum

def epsilon_greedy(epsilon, state,model,train=False):

    if train or np.random.rand() < ((epsilon / 4) + (1 - epsilon)):
        action = tf.argmax(model(encode(4,state),1))
    else:
        # need to be torch tensor
        action = tf.Variable(np.random.randint(0,4))

    return action



gamma = 0.999  # 0.999
epsilon = 1.
episodes = 500
max_steps = 250  # 2500

env = gym.make('FrozenLake-v0',is_slippery=False)
batch_size = 5
target_update_counter = 0
iter_index = []
iter_reward = []
iter_total_steps = []
timestep_reward = []
memory = ReplayMemory(80)
model = VQC()
for episode in range(episodes):
    print(f"Episode: {episode}")
    s = env.reset()
    a = int(epsilon_greedy(epsilon,s,model))
    t = 0
    total_reward = 0
    done = False

    while t < max_steps:
        print(t)
        t += 1
        target_update_counter += 1
        s_, reward, done, info = env.step(a)
        total_reward += reward
        a_ = int(epsilon_greedy(epsilon,s,model))
        memory.push(s, a, reward, s_, done)

        if len(memory) > batch_size:
            batch = memory.sample(batch_size=batch_size)
            model.train_step(batch)

        s, a = s_, a_

        if done:
            epsilon = epsilon / ((episode / 100) + 1)
            print(f"This episode took {t} timesteps and reward: {total_reward}")
            timestep_reward.append(total_reward)
            iter_index.append(episode)
            iter_reward.append(total_reward)
            iter_total_steps.append(t)
            break

print(timestep_reward)