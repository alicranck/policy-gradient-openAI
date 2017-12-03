'''almog elharar - 203549407, Yair Baruch - 043463256
parts of the code were adapted from: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb'''


import gym
from gym import wrappers
import numpy as np
import cPickle as pickle
import os.path
import tensorflow as tf
os.path.isfile

env_d = 'LunarLander-v2'
env = gym.make(env_d)
env.reset()

# define constants and collect env metadata
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
batch_size = 20
total_episodes = 1500
discount_factor = 0.99
learning_rate = 0.005
hidden_units = 30

# placeholders for keeping episode
observation = tf.placeholder(tf.float32, shape=[None,state_dim])
action = tf.placeholder(tf.float32, shape=[None,num_actions])
probabilities = tf.placeholder(tf.float32, shape=[None, num_actions])
accum_reward = tf.placeholder(tf.float32, shape=[None])

#in case there are saved weights
fd = 'ws.p'


def agent(obsrv):                             # the policy network graph
    with tf.variable_scope("policy_network") :
        w1 = tf.get_variable("W1",[state_dim, hidden_units] ,initializer=tf.contrib.layers.xavier_initializer())  #
        b1 = tf.get_variable("b1",[hidden_units] ,initializer=tf.contrib.layers.xavier_initializer())  		  #   layer 1
        h1 = tf.tanh(tf.add(tf.matmul(obsrv, w1), b1))                                                 		  #

        w2 = tf.get_variable("W2",[hidden_units, hidden_units] ,initializer=tf.contrib.layers.xavier_initializer())  #
        b2 = tf.get_variable("b2",[hidden_units] ,initializer=tf.contrib.layers.xavier_initializer())  		     #   layer 2
        h2 = tf.tanh(tf.add(tf.matmul(h1, w2), b2))              				       		     #

        w3 = tf.get_variable("W3",[hidden_units, num_actions] ,initializer=tf.contrib.layers.xavier_initializer())  #
        b3 = tf.get_variable("b3",[num_actions] ,initializer=tf.contrib.layers.xavier_initializer())      	    #   output layer
        y = tf.nn.softmax(tf.add(tf.matmul(h2, w3), b3))							    #
      
    return y

probabilities = agent(observation)


# Generate holders for all trainable variables in the model
tvars = tf.trainable_variables()
gradient_holders = []
for i, var in enumerate(tvars):
        gradient_holders.append(tf.placeholder(tf.float32,name=str(i)+'_holder'))


# Define the optimizer and apply costume gradients
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.apply_gradients(zip(gradient_holders, tvars))


# Compute gradients of the log probabilities
responsible_probs = tf.reduce_sum(tf.multiply(probabilities, action), reduction_indices=[1])
loss = -tf.reduce_mean(tf.log(responsible_probs)*accum_reward)
gradients = tf.gradients(loss, tvars)


#placeholders for saving the variables
parameters_assign = []
parameters_W = tf.placeholder(tf.float32, shape=[None,None])
parameters_b = tf.placeholder(tf.float32, shape=[None])
for i,var in enumerate(tvars):
    if i % 2 ==0:
        parameters_assign.append(var.assign(parameters_W))
    else:
        parameters_assign.append(var.assign(parameters_b))


# Runs a complete episode and returns probabilities, action, and reward for each step
def run_episode(agent, sess, env):
    states = []
    probs = []
    rewards = []
    actions = []
    obsrv = env.reset()
    cumulative_reward = 0
    while (True):
        obsrv_vector = np.expand_dims(obsrv, axis=0)
        states.append(obsrv_vector)
        # Run the policy network and get a distribution over actions
        action_probs = np.float64(sess.run(agent, feed_dict={observation:obsrv_vector}))
	#print(action_probs)

	action_probs = np.divide(action_probs,np.sum(action_probs))
	#print(action_probs)
        # Sample action from distribution
        action = np.random.multinomial(1, np.squeeze(action_probs))
        # Step the environment and get new measurements
        obsrv, reward, done, info = env.step(np.argmax(action))
        probs.append(action_probs)
        rewards.append(reward)
        actions.append(action)
        cumulative_reward += reward
        if done:
            break
    return states, probs, actions, rewards, cumulative_reward

#take 1D float array of rewards and compute discounted reward 
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * discount_factor + r[t]
        discounted_r[t] = running_add
    return discounted_r


#-------------------------------------------------training loop--------------------------------------------------------------------

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def main(argv):
    with tf.Session() as sess:
	sess.run(init)
        episode_number = 0

	#load saved variables - optional
	if os.path.isfile(fd):		
            params = pickle.load(open(fd,'r'))
            for i,var in enumerate(params):
            	if i % 2 ==0:
                    sess.run(parameters_assign[i], feed_dict={parameters_W:var})
            	else:
                    sess.run(parameters_assign[i], feed_dict={parameters_b:var})

	# Define a buffer to hold gradients of all episodes in a batch
        gradBuffer = sess.run(tf.trainable_variables())  
        avg_reward = 0
        while episode_number <= total_episodes:

	    #reset gradient buffer
            for ix,item in enumerate(gradBuffer):
                   gradBuffer[ix] = item * 0
            
            for i in range(batch_size):
                grad_list = []

                # run episode and collect all information 
                ep_states ,ep_probs, ep_actions, ep_rewards, ep_cumulreward = run_episode(probabilities, sess, env)
                avg_reward += ep_cumulreward/batch_size

                # process rewards
                ep_rewards = discount_rewards(np.array(ep_rewards))
                ep_rewards -= np.mean(ep_rewards)
                ep_rewards /= np.std(ep_rewards)

                # calculate loss
                ep_states = np.vstack(ep_states)
                ep_probs = np.vstack(ep_probs)
                grad_list = sess.run(gradients, feed_dict={observation:ep_states, action:ep_actions, probabilities:ep_probs, accum_reward:ep_rewards})
                for idx, item in enumerate(grad_list):
                        gradBuffer[idx] += item*(1.0/batch_size)

            # update model
            sess.run(train_step, feed_dict=dict(zip(gradient_holders, gradBuffer)))
            episode_number += 1 
            # Save model every 10 updates
            if episode_number%10==0:
                with open('ws.p', 'wb') as paramsFile:
                    with tf.variable_scope("policy_network", reuse=True):
                        W1 = tf.get_variable("W1",[state_dim, hidden_units]).eval()  
                        b1 = tf.get_variable("b1",[hidden_units]).eval()
                        W2 = tf.get_variable("W2",[hidden_units, hidden_units]).eval()
                        b2 = tf.get_variable("b2",[hidden_units]).eval()
                        W3 = tf.get_variable("W3",[hidden_units, num_actions]).eval()
                        b3 = tf.get_variable("b3",[num_actions]).eval()
                        pickle.dump([W1,b1,W2,b2,W3,b3], paramsFile)
                print("avg reward for 10 iterations: "+ str(avg_reward/10))
                avg_reward=0

if __name__ == '__main__':
    tf.app.run()
