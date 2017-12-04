import test.unit
import tranding.q

# data has to have: pct change, actions, cummulative
actions = ['buy', 'sell', 'noop']
def reward_function(state, action, meta):
    # Have to sell within 10 days of the first buy
    # buying = 1
    # nothing = 0 + cummulative change
    # selling = cummulative change * 2
    pass

def meta_update(state, meta, action):
    # calculate cummulative return
    pass

learner = trading.q.Learner()

learner.data = test.unit.data
learner.reward_function = reward_function
# alpha is learning rate
learner.learn(iterations=100, alpha=0.2, gamma=0.2)

meta = []
for datum in test.unit.data:
    # history = [{state, action, meta},...]
    meta = learner.get_action(datum, meta, meta_update)

def calculate_return(history):
    # sum up cummulative return
    pass
