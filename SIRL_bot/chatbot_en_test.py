from utils import *
from deep_dialog.agents.agent import Agent
from deep_dialog.nlu.q_table import Q_table as Q_tab
from evaluate import *
from statsmodels.stats.proportion import proportion_confint
import os
import copy
import pandas as pd
import pickle

def episodes_run(agent, NLU, simulator, model, params, mapping, output='epochs.csv', use_model = True, thre_no_intent=0.5):
    """ Run episodes and train DQN 
    Parameters:
    -------------------------------
    agent: instance of class Agent
    NLU: instance of class NLU
    simulator: instance of class Simulator
    model: instance of class score_model
    params: dictionary (RL agent configuration params)
    mapping: dict (maps from index to set of states/action and viceversa)
    output: string (optional)
    use_model: boolean (optional)
         if True use score model as rewards
         if False use interactive rewards (prompt the user)
    thre_no_intent: float (optional)     
    """

    ### Configure RL agent
    num_episodes = params['num_episodes']
    train_freq = params['train_freq']

    map_state2index = mapping['state2index']
    map_index2state = mapping['index2state']
    map_index2action = mapping['index2action']
    map_action2index = mapping['action2index']

    map_action2answer = pd.read_csv('intent_response.csv')
    num_states = len(map_index2state)


    ## Note for improvement: this computation can be done at an higher level and feed also the score_model
    emb_states = all_embeddings(map_index2state)
    emb_actions  = all_embeddings(map_index2action,  map_action = map_action2answer)

    # init evaluator
    eval = evaluate(map_state2index, map_index2action)
        
    sum_rewards = 0
    sum_rewards_bin = 0
    count_episodes = 0
    summary = []
    epoch = 50
    #episode_results = []
    #rewards_bin_episode = []
    #rewards_episode =[]

    flush = False
    
    while True:
        print('----------------------------------------')
        utterance = input("You: ")
        if utterance.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        map_state2index = mapping['state2index']
        if utterance in map_state2index:
            # get one-hot representation
            repr = get_representation(utterance, map_state2index)

            # get action index from agent running epsilon-greedy policy
            index_action = agent.run_policy(repr, epoch)

            # convert index to action
            action = map_index2action[index_action]
            print('chatbot_predict_intent:', action)
            if index_action in map_index2action:
                answers = [map_action2answer.loc[map_action2answer['intent'] == action]['response'].values[0]]
                print('chatbot_answer: ', answers)
            
        else:
            print('I have no idea about your question')
            
    return

def get_representation(utterance, map_state2index):
    """ return one-hot representation of the given sentence """
    
    size = len(map_state2index)
    index = map_state2index[utterance]
    rep = np.zeros((1,size))
    rep[0, index] = 1.0

    return rep

def all_embeddings(map_index2sent, map_action=None):
    """ compute embeddings for all states or actions """
    
    indices, sentences = zip(*map_index2sent.items())

    if map_action is not None:
        
         answers = [map_action.loc[map_action['intent'] == action]['response'].values[0] for action in sentences]
         sentences = answers
    
    EMB = embeddings([], embed_par='tf')
    emb = EMB.fit(sentences)

    return dict(zip(indices, emb))


def testing(agent, mapping, Q_table, from_Q_table = True, verbose=True, num_test=-1):
    """ Test performance of DQN agent after warm-up training, by comparing responses with NLU responses"""
    
    map_state2index = mapping['state2index']
    map_index2state = mapping['index2state']
    map_index2action = mapping['index2action']
    map_action2index = mapping['action2index']


    if num_test < 0 :
        indices = range(agent.user_act_cardinality)
    else:
        indices = range(num_test)
        

    num_success = 0
    for index in indices:

        example = map_index2state[index]
        rep = np.zeros((1,len(map_index2state)))
        rep[0, index] = 1.0

        index_action_dqn = agent.dqn.predict(rep, {'gamma': agent.gamma})

        if from_Q_table:
            index_watson, _ = Q_table.predict(index)
            action_watson = map_index2action[index_watson]
        else:
            action_watson,_,_ = agent.rule_policy(example)
            index_watson = map_action2index[action_watson]

        if index_watson - index_action_dqn == 0:
            num_success += 1

        if verbose:    
            print('------------------------------------------------------')
            print('input:', example, index)
            print('action NLU:', action_watson, index_watson)
            print('action DQN:', map_index2action[index_action_dqn])
            print('------------------------------------------------------')

    print('In warm-up: # success %d, success rate %f' %( num_success, float(num_success)/float(len(indices))))

    return
        
import os
import copy
import pandas as pd
import pickle
from utils import *
from deep_dialog.qlearning import DQN
from deep_dialog.nlu.rasa import rasa
from deep_dialog.nlu.rasa import Interpreter
from learning_scores.score_model import *
from deep_dialog.agents.agent import Agent

def set_nlu_model(params, top_intent):
    rasa_model_path = './projects/default/nlu_20240224-014142_checkpoint'
    rasa_NLU = rasa(params, top_intent, isTest=True)
    rasa_NLU.cl_threshold = 0.1
    rasa_NLU.no_intent = 'No intent detected'
    rasa_NLU.interpreter = Interpreter.load(rasa_model_path)
    
    return rasa_NLU

params = parameters()
df, dict_intents, intents = load_data(params, max_intents = 6)
NLU = set_nlu_model(params, intents)

mapping = get_mapping(NLU, df)
# print(mapping)

simulator = Simulator(df, multiply=4)

agent = Agent(NLU, params, mapping, warm_start=1, model_path='./models/agt_9.p')


params_model = config_score_model()
model = score_model(params_model)
print(model.path_to_model)
model.path_to_model = './trained_model'
print(model.path_to_model)

results = episodes_run(agent, None, simulator, model,  params, mapping)