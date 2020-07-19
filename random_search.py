
import numpy as np
import logging
import random
random.seed(59)

from aug_search import AUG_TYPE
from reward_calc_imgnet import RewardCal
from tqdm import tqdm
import pdb


# Search space hyper parameters
OP_NUM = 5
TYPE_SPACE = len(AUG_TYPE.keys())
WEIGHT_SPACE = 10
PROB_SPACE = 10
RANGE_SPACE = 10


# Search process hyper parameters
interval = 0.2
sample_batch = 4
lr = 0.1
epoch = 300

RESUME = True
resume = 210
policy = [[2.0, 7.0, 8.0, 8.0],
          [3.0, 9.0, 7.0, 7.0],
          [1.0, 9.0, 7.0, 9.0],
          [4.0, 9.0, 9.0, 8.0],
          [2.0, 8.0, 9.0, 9.0]]
best_policy = [[2.0, 7.0, 8.0, 8.0],
               [3.0, 9.0, 8.0, 7.0],
               [1.0, 9.0, 8.0, 9.0],
               [4.0, 9.0, 9.0, 8.0],
               [2.0, 8.0, 9.0, 9.0]]
best_reward =  13.447103881835938



# A policy could be represent by a 5x4 list,
# [[type, weight, prob, range], ...]

def random_policy():

    policy = []
    for i in range(OP_NUM):
        type_ = random.randint(0, TYPE_SPACE-1)
        weight_ = random.randint(0, WEIGHT_SPACE-1)
        prob_ = random.randint(0, PROB_SPACE-1)
        range_ = random.randint(0, RANGE_SPACE-1)

        policy.append([type_, weight_, prob_, range_])

    return policy


def restrict(policy):
    ''' Input：  np array [5, 4]
        Output： np array [5, 4]
    '''
    policy = np.around(policy)
    return np.clip(policy, 0, [TYPE_SPACE-1, WEIGHT_SPACE-1, PROB_SPACE-1, RANGE_SPACE-1])


def single_epoch(policy, reward_getter, lr=0.1, sample_batch=10):

    deltas = []
    rewards = []

    best_reward = 0

    reward_getter.randomrize_models()

    policy_update = np.array(policy)

    # Sample deltas and update policy
    for i in range(sample_batch):
        sample_delta = random_policy()

        policy_plus  = restrict((np.array(policy) + np.array(sample_delta)*interval)).tolist()
        policy_minus = restrict((np.array(policy) - np.array(sample_delta)*interval)).tolist()

        reward_plus = reward_getter.get_reward(policy_plus)
        reward_minus = reward_getter.get_reward(policy_minus)

        policy_update = policy_update + lr/sample_batch * (reward_plus - reward_minus) \
                                                        * np.array(sample_delta)

        # Policy with highest reward may appear here.
        if reward_plus > best_reward:
            best_reward = reward_plus
            best_policy = policy_plus
        if reward_minus > best_reward:
            best_reward = reward_minus
            best_policy = policy_minus


    # Test updated policy reward
    updated_policy = restrict(policy_update).tolist()
    updated_reward = reward_getter.get_reward(updated_policy)

    return updated_policy, updated_reward, best_policy, best_reward


if __name__ == '__main__':



    # Initialize
    if not RESUME:
        best_reward = 0
        policy = random_policy()

    reward_getter = RewardCal()

    logger = logging.getLogger('controller')
    hdlr = logging.FileHandler('./log/random_search.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    for i in tqdm(range(resume+1, epoch)):
        policy, reward, epoch_best_policy, epoch_best_reward = single_epoch(policy, reward_getter,
                                                                            lr=lr, sample_batch=sample_batch)

        if reward > best_reward:
            best_reward = reward
            best_policy = policy

        if epoch_best_reward > best_reward:
            best_reward = epoch_best_reward
            best_policy = epoch_best_policy

        logger.info("Iter {}".format(i))
        logger.info("Policy now: {}".format(policy))
        logger.info("Reward now: {}".format(reward))
        logger.info("Best policy so far: {}".format(best_policy))
        logger.info("Best reward so far: {}".format(best_reward))












