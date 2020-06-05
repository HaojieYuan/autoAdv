from aug_search import AUG_TYPE

def log(logger, iteration, actions, reward):
    logger.info("iter {}".format(iteration))
    for i in range(actions.shape[0]):
        policy = ''
        for j in range(actions.shape[1]):
            policy = policy + '['
            policy = policy + AUG_TYPE[actions[i,j,0,0].detach().cpu().item()] + ':'
            policy = policy + str(actions[i,j,0,1].detach().cpu().item()) + ', '
            policy = policy + AUG_TYPE[actions[i,j,1,0].detach().cpu().item()] + ':'
            policy = policy + str(actions[i,j,1,1].detach().cpu().item())
            policy = policy + '] '
        logger.info("Policy {}: {}".format(i+1, policy))
        logger.info("reward: {}".format(reward[i,0].detach().cpu().item()))
