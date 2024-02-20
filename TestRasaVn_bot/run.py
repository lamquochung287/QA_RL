from utils import *
#import watson_developer_cloud
from deep_dialog.agents.agent import Agent
from deep_dialog.nlu.nlu import set_nlu
from deep_dialog.nlu.q_table import Q_table as Q_tab
# from deep_dialog.nlu.watson import  watson
import os
from learning_scores.embeddings import *
from learning_scores.score_model import *
from statsmodels.stats.proportion import multinomial_proportions_confint
from episodes import *
import pandas as pd

import time
import datetime
from datetime import datetime as dt
start = time.perf_counter()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_path=None #use model file if you want to start from pre-trained model. 

######################################################
# load configuration
######################################################
params = parameters()

#######################################################
# load logs data --> used by the simulator
######################################################
df, dict_intents, intents = load_data(params, max_intents = 6)

#######################################################
# Load NLU agent - rasa or watson 
######################################################
NLU = set_nlu(params, 'rasa', intents)

# return dict with mapping of sets of states/actions to index
mapping = get_mapping(NLU, df) #defined in utils.py

#######################################################
# Load user Simulator (defined in utils.py)
######################################################
simulator = Simulator(df, multiply=4)


#######################################################
# Init Agent (defined in deep_dialog/agents/agent.py)
######################################################
warm_start = 1
agent = Agent(NLU, params, mapping, warm_start, model_path=None)


#######################################################
#  WARM-UP phase (warmup_run defined in episodes.py)
######################################################
if model_path is None:

        # train Q in warm-up phase --> reward from Watson CI. Set use_Q to True if you want the NLU resp to be taken from Q table (pickle file already created)
        agent, Q_table = warmup_run(agent, simulator, params, mapping, use_Q=False, verbose=False)

        # test DQN accuracy by cfr DQN with NLU results
        testing(agent, mapping, Q_table, verbose=False)
        print('... END of WARM-UP PHASE')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


#######################################################
#  Train score model
######################################################

# load score mdoel configuration (in utils.py)
params_model = config_score_model()

# load dat to train score model (utils.py)
x, y, emb_u, emb_r = score_data(params)

#init score model (in learning_scores/score_model.py)
model = score_model(params_model)

print('Training score model...')
Final_train_acc, _ = model.fit(x,y, emb_u=emb_u, emb_r=emb_r, path_to_model = "./trained_model/model_final.ckpt")

print('Train Accuracy of score model:', Final_train_acc)
print('----------------------------------------------------')


#######################################################
#  Run episodes
######################################################
print('---------- Running RL episodes -----------')
results = episodes_run(agent, NLU, simulator, model,  params, mapping)

print('End of episodes !')
print()
results.to_csv('summary.csv', index=False)

end = time.perf_counter()
second_final = end - start
time_running = datetime.timedelta(seconds=second_final)
print(time_running)

time_running_obj = dt.strptime(time_running, "%H:%M:%S.%f")
params['time_running'] = f'{str(time_running_obj.hour)}h'

a = params['time_running']
b1 = params['numInterOfAgentTrainWarmup_batchSize']
b2 = params['numInterOfAgentTrainWarmup_numBatches']
b3 = params['numInterOfAgentTrainWarmup_numIter']
b4 = 'T' if params['numInterOfAgentTrainWarmup_miniBatches'] == True else 'F'
c1 = params['numInterOfAgentDQNTraining_batchSize']
c2 = params['numInterOfAgentDQNTraining_numBatches']
c3 = params['numInterOfAgentDQNTraining_numIter']
c4 =  'T' if params['numInterOfAgentDQNTraining_miniBatches'] == True else 'F'
d = params['dqn_hidden_size']
e = params['num_episodes']
f = params['buffer_size']

results.to_csv(f'summary_{a}_{b1}{b2}{b3}{b4}_{c1}{c2}{c3}{c4}_{d}_{e}_{f}.csv', index=False)
