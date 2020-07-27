
import numpy as np
import logging
import random

from aug_search import AUG_TYPE
from reward_calc_imgnet import RewardCal
from tqdm import tqdm
import argparse
import os
import pdb


parser = argparse.ArgumentParser(description='Search pipeline.')
parser.add_argument('--opnum', default=5, type=int,
                    help=' num of aug branch specified.')
parser.add_argument('--opdepth', default=2, type=int,
                    help=' num of aug on each branch. ')
parser.add_argument('--policybatch', default=4, type=int,
                    help=' policy num sampled each batch. ')
parser.add_argument('--attackbatch', default=8, type=int,
                    help=' attack batch size. ')
parser.add_argument('--datasplit', default=500, type=int,
                    help=' peice num of Imagenet dataset should be splited. ')
parser.add_argument('--log', default='random_search.log', type=str,
                    help=' log file saved in log dir. ')


# Search space hyper parameters
ARGS = parser.parse_args()
OP_NUM = ARGS.opnum
OP_DEPTH = ARGS.opdepth
TYPE_SPACE = len(AUG_TYPE.keys())
WEIGHT_SPACE = 10
PROB_SPACE = 10
RANGE_SPACE = 10


# Search process hyper parameters
interval = 0.3
sample_batch = ARGS.policybatch
lr = 0.1
epoch = 300

RESUME = False
resume = 210
policy = None
best_policy = None
best_reward =  13.447103881835938



#           # different operations on same branch
# [[weight, [type, prob, range], [type, prob, range]...], # different branch
#  [weight, [type, prob, range], [type, prob, range]...], # different branch
#  ...]

def random_policy():
    ''' Sample a policy randomly. '''
    policy = []
    for i in range(OP_NUM):
        branch = []

        weight_ = random.randint(0, WEIGHT_SPACE-1)
        branch.append(weight_)

        for j in  range(OP_DEPTH):
            type_ = random.randint(0, TYPE_SPACE-1)
            prob_ = random.randint(0, PROB_SPACE-1)
            range_ = random.randint(0, RANGE_SPACE-1)

            branch.append([type_, prob_, range_])

        policy.append(branch)

    return policy


def restrict(policy):
    ''' Input：  policy defined by our algorithm
        Output： restricted policy
    '''
    #policy = np.around(policy)
    #return np.clip(policy, 0, [TYPE_SPACE-1, WEIGHT_SPACE-1, PROB_SPACE-1, RANGE_SPACE-1])
    restricted_policy = []

    for branch in policy:
        restricted_branch = []
        weight_ = udf_clip(branch[0], 0, WEIGHT_SPACE-1)
        restricted_branch.append(weight_)

        for j in range(1, len(branch)):
            type_ = udf_clip(branch[j][0], 0, TYPE_SPACE-1)
            prob_ = udf_clip(branch[j][1], 0, PROB_SPACE-1)
            range_ = udf_clip(branch[j][2], 0, RANGE_SPACE-1)

            restricted_branch.append([type_, prob_, range_])

        restricted_policy.append(restricted_branch)

    return restricted_policy


def update_policy(policy, direction, interval):
    updated_policy = []

    for branch1, branch2 in zip(policy, direction):
        updated_branch = []
        weight_ = round(branch1[0] + interval*branch2[0])
        updated_branch.append(weight_)

        for j in range(1, len(branch1)):
            type_ = round(branch1[j][0] + interval*branch2[j][0])
            prob_ = round(branch1[j][1] + interval*branch2[j][1])
            range_ = round(branch1[j][2] + interval*branch2[j][2])

            updated_branch.append([type_, prob_, range_])

        updated_policy.append(updated_branch)

    return updated_policy


def remove_duplicate(policy):
    out_policy = []
    for branch in policy:
        out_branch = []
        out_branch.append(branch[0])

        all_types = []
        for j in range(1, len(branch)):
            all_types.append(branch[j][0])

        if len(all_types) == len(set(all_types)):
            for j in range(1, len(branch)):
                out_branch.append(branch[j])
        else:
            all_types_no_dul = []
            for j in range(len(all_types)):
                type_now = all_types[j]
                while type_now in all_types_no_dul:
                    type_now = random.randint(0, TYPE_SPACE-1)
                all_types_no_dul.append(type_now)

            for j in range(1, len(branch)):
                out_branch.append([all_types_no_dul[j-1],
                                   branch[j][1], branch[j][2]])

        out_policy.append(out_branch)

    return out_policy


def udf_clip(input_, bound_low, bound_high):
    output_ = bound_low if input_ < bound_low else input_
    output_ = bound_high if output_ > bound_high else output_

    return output_



def single_epoch(policy, reward_getter, lr=0.1, sample_batch=10):

    deltas = []
    rewards = []

    best_reward = 0

    reward_getter.randomrize_models()

    policy_update = policy.copy()

    # Sample deltas and update policy
    for i in range(sample_batch):
        sample_delta = random_policy()

        policy_plus  = restrict(update_policy(policy, sample_delta, interval))
        policy_minus = restrict(update_policy(policy, sample_delta, -interval))

        policy_plus = remove_duplicate(policy_plus)
        policy_minus = remove_duplicate(policy_minus)

        reward_plus = reward_getter.get_reward(policy_plus,
                                               batch_size=ARGS.attackbatch,
                                               dataset_split=ARGS.datasplit)
        reward_minus = reward_getter.get_reward(policy_minus,
                                                batch_size=ARGS.attackbatch,
                                                dataset_split=ARGS.datasplit)

        # Remove duplicate may change delta, so we recalculate delta here.
        true_delta = update_policy(policy_plus, policy_minus, -1) #policy_plus-policy_minus

        # The actual delta should be policy_plus - policy
        # so we have 0.5*lr here.
        policy_update = update_policy(policy_update, true_delta,
                                      0.5*lr/sample_batch * (reward_plus-reward_minus))


        # Policy with highest reward may appear here.
        if reward_plus > best_reward:
            best_reward = reward_plus
            best_policy = policy_plus
        if reward_minus > best_reward:
            best_reward = reward_minus
            best_policy = policy_minus


    # Test updated policy reward
    updated_policy = restrict(policy_update)
    updated_policy = remove_duplicate(updated_policy)
    updated_reward = reward_getter.get_reward(updated_policy)

    return updated_policy, updated_reward, best_policy, best_reward


if __name__ == '__main__':

    # Initialize
    if not RESUME:
        best_reward = 0
        policy = random_policy()
        resume = -1

    reward_getter = RewardCal()

    logger = logging.getLogger('controller')
    hdlr = logging.FileHandler(os.path.join('./log', ARGS.log))
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












