# import test.unit
# import tranding.q
#
# # data has to have: pct change, actions, cummulative
# actions = ['buy', 'sell', None]
# def reward_function(state, action, meta):
#     if action == 'buy':
#         return 1
#     elif action == None and meta.get('days') > 0:
#         return meta['cummulative'] / 2.0
#     elif action == 'sell':
#         return 2.0 * meta['cummulative']
#
# # meta will have
# # - # of days
# # - cummulative
# def meta_update(state, meta, action):
#     if action == 'buy' and meta.get('days') == 0:
#         meta['days'] = 1
#     # calculate cummulative return
#     pass
#
# learner = trading.q.Learner()
#
# learner.data = test.unit.data
# learner.reward_function = reward_function
# learner.meta_update = meta_update
# # alpha is learning rate
# # gamma is discount factor
# learner.learn(iterations=100, alpha=0.2, gamma=0.8)
#
# history = learner.run(test.unit.data)
#
# def calculate_return(history):
#     # sum up cummulative return
#     pass
