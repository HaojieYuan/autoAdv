from controller import Controller
import torch
from reward_calc import get_rewards
import numpy as np
import random
from tqdm import tqdm
import logging
from utils import log
import pdb

# random control
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

# weights
type_pg_weight = 1
magnitude_pg_weight = 0.5
type_entropy_weight = 0.1
magnitude_entropy_weight = 0.1
baseline_decay = 0.9
learning_rate = 1e-5

# train hypher params
batch_size = 5
iterations = 1000

# Netwrok hypher params
hid_size = 1000
ckpt_path = './controller_best.ckpt'


# Initilaize
baseline = None
model = Controller(hid_size).cuda(0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Logger
logger = logging.getLogger('controller')
hdlr = logging.FileHandler('./log/first_run.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

# Loop.
best_reward = 0
for iteration in tqdm(range(iterations)):
    actions, log_probs, entropies = model.sample(batch_size) 

    #reward = torch.randn(batch_size, 1).cuda(0)
    reward = get_rewards(actions)
    log(logger, iteration, actions, reward)

    reward_sum = torch.sum(reward)
    if reward_sum > best_reward:
        best_reward = reward_sum
        torch.save(model.state_dict(), ckpt_path)


    # Loss Calculation.
    reward = reward + type_entropy_weight * entropies['type'] + \
                    magnitude_entropy_weight * entropies['magnitude']
    if baseline is None:
        baseline = reward
    else:
        baseline = baseline_decay*baseline.detach() + (1-baseline_decay)*reward
    adv = reward - baseline
    loss = -adv * (type_pg_weight*log_probs['type'].reshape(batch_size, -1) + \
                magnitude_pg_weight*log_probs['magnitude'].reshape(batch_size, -1))
    loss = loss.sum()

    # BP.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()