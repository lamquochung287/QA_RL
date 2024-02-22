import pandas as pd
import random
import numpy as np
import copy
import json
from learning_scores.embeddings import *
from pathlib import Path

def parameters():
    """ configuration for RL agent """
    
    config = {}
    
    config['data_path'] = './data/main'
    config['data_file'] = [
        'chapter01.csv','chapter02.csv','chapter03.csv',
        # 'chapter04.csv','chapter05.csv','chapter06.csv',
        # 'chapter07.csv','chapter08.csv','chapter09.csv',
        ]
    
    config['training_path'] = './data/training'
    config['training_path_part1'] = './data/training/part_1'
    config['training_path_part2'] = './data/training/part_2'
    config['training_path_part3'] = './data/training/part_3'
    config['training_file'] = [
        'chapter01_data_training.json','chapter02_data_training.json','chapter03_data_training.json',
        # 'chapter04_data_training.json','chapter05_data_training.json','chapter06_data_training.json',
        # 'chapter07_data_training.json','chapter08_data_training.json','chapter09_data_training.json',
        ]
    
    config['intent_response_path'] = './data/intent_response'
    config['intent_response_file'] = [
        'chapter01_intent_response.csv','chapter02_intent_response.csv','chapter03_intent_response.csv',
        # 'chapter04_intent_response.csv','chapter05_intent_response.csv','chapter06_intent_response.csv',
        # 'chapter07_intent_response.csv','chapter08_intent_response.csv','chapter09_intent_response.csv',
        ]
    
    config['test_path'] = './data/test'
    config['test_file'] = [
        'chapter01_test_set.csv','chapter02_test_set.csv','chapter03_test_set.csv',
        # 'chapter04_test_set.csv','chapter05_test_set.csv','chapter06_test_set.csv',
        # 'chapter07_test_set.csv','chapter08_test_set.csv','chapter09_test_set.csv',
        ]
    
    config['nlu'] = 'rasa' #either watson or rasa
    config['num_episodes_warm'] = 10 # should be similar to number of conversations 
    config['train_freq_warm'] = 10 # if equal to num_episodes_warm --> 1 training epoch, which is enough for warmp-up
    # config['num_episodes'] = 50 # total number of episodes 
    config['train_freq'] =  10 #30 number of episodes per training epochs
    config['epsilon'] = 0.2 #initial
    config['epsilon_f'] = 0.05 #epsilon after epsilon_epoch_f epochs
    config['epsilon_epoch_f'] = 20  
    # config['dqn_hidden_size'] = 64 #paramter of DQN
    config['gamma'] = 0.9 #paramter of DQN 
    config['prob_no_intent'] = 0.2 #probability of giving 'No intent detected" in random policy
    # config['buffer_size'] = 20000 # max size of experience replay buffer 
    
    # some config to set special params to save sumary's title file for more detail
    # structure:
    #   sumary_timeRunning_numInterOfAgentTrainWarmup_numInterOfAgentDQNTraining_dqnHiddenSize_numOfEpisodes_bufferSize
    # numInterOfAgentTrainWarmup
    config['numInterOfAgentTrainWarmup_batchSize'] = 4
    config['numInterOfAgentTrainWarmup_numBatches'] = 10
    config['numInterOfAgentTrainWarmup_numIter'] = 10
    config['numInterOfAgentTrainWarmup_miniBatches'] = False
    # numInterOfAgentDQNTraining 
    config['numInterOfAgentDQNTraining_batchSize'] = 10
    config['numInterOfAgentDQNTraining_numBatches'] = 30
    config['numInterOfAgentDQNTraining_numIter'] = 20
    config['numInterOfAgentDQNTraining_miniBatches'] = False
    # dqnHiddenSize
    config['dqn_hidden_size'] = 64
    # numOfEpisodes
    config['num_episodes'] = 100
    # bufferSize
    config['buffer_size'] = 20000
    
    return config

def config_score_model():
    """ configuration for score model """
    
    params = {}
    params['epochs'] = 40 # when do 50 epochs, in epoch 10-20-30-40 have same as loss and train acc, so try 30 is fit
    params['lr'] = 1.e-4
    params['l2_reg'] = 0.1
    params['print_freq'] = 10

    return params

def watson_config():
    """ set configuration to connect to Watson workspace ID if NLU=Watson """
    
    username = ''
    password =''
    WID = ''

    return

def multiple_data_location(params, file_path, file_type):
    files = []
    prefix = params[file_path]
    
    for fileNames in params[file_type]:
        path = Path(prefix + '/' + fileNames)
        file_exists = path.is_file()
        if file_exists == True: 
            files.append(prefix + '/' + fileNames)
    
    return files

def read_multiple_files(files, suffix='csv'):
    # df = pd.read_csv(file)
    _list = []
    if suffix == 'json':
        for filename in files:
            with open(filename, 'r', encoding='utf-8') as f:
                js = json.load(f)
                _list.append(js)
        
    if suffix == 'csv':
        dfs = []
        for filename in files:
            df = pd.read_csv(filename)
            dfs.append(df)
        _list = pd.concat(dfs, ignore_index=True)
    
    return _list

def load_data(params, max_intents = 7 ):    
    """ Load dataset with conversation already transformed in conventient format: question, answer, reward, feedback, along with intents counter dict and list of top intents """

    print('------------- Loading conversations ---------------')

    # file = params['log_file']
    df_location = multiple_data_location(params, file_path='data_path', file_type='data_file')
    df = read_multiple_files(df_location)
    
    df.dropna(subset=['reward'], how='all', inplace = True)
    df.dropna(subset=['question'], how='all', inplace = True)
    df.dropna(subset=['answer'], how='all', inplace = True)

    print('lenght of original DataFrame:', len(df.index))
    
    df_intents, dict_intent, intents = get_intents(df, max_num_intents = max_intents)
    print('Size of intents dataframe:', len(df_intents.index))
    
    # remove feedback intent, as it does not have answer
    df_intents = df_intents[df_intents['intent'] != 'Feedback']

    print()
    
    return df_intents, dict_intent, intents

# def get_df_feedback(df):
#     """ transform original dataset of covnewrsation into convenient format with feedback """
    
#     indices = []
#     feedbacks = []
#     count_feedback = 0

#     for index, data in df.iterrows():

#         utt_feedback =['no', 'yes']
#         utt = data['question']
#         intent = data['intent']
#         reward = data['reward']

#         utt = utt.lower()

#         indices.append(index)

#         is_feedback = any([utt.startswith(f) for f in utt_feedback])

#         if intent.lower() == 'feedback':

#             feedback = 0 if utt.startswith('yes') else 1

#             feedbacks.append({'question': previous['question'], 'answer': previous['answer'],
#             'feedback':feedback, 'reward':previous['reward'], 'intent':previous['intent']})

#             count_feedback += 1

#         previous = data   

#     print('Number of feedbacks:', count_feedback)
#     f = [item['feedback'] for item in feedbacks]
#     print('positive:',f.count(0))
#     print('negative:',f.count(1))

#     return pd.DataFrame(feedbacks)



def get_intents(df, max_num_intents = 90, remove_fallback=True):
    """ This function selects the max_num_intents intents more frequently triggered by the users and return dataframe with these intents only"""

    # df_feedback = get_df_feedback(df)

    intents = list(df['intent'])
    unique_intents = list(set(intents))
    print('number of intents triggered:', len(unique_intents))

    count_intent = []

    for intent in unique_intents:

        dict_intent = {}
        dict_intent['intent'] = intent
        dict_intent['counts'] = intents.count(intent)
        count_intent.append(dict_intent)

    counts = np.array([dic['counts'] for dic in count_intent])
    isort = np.argsort(counts)[::-1]

    count_intent_sorted = np.array(count_intent)[isort]

    intents = [dictionary['intent'] for dictionary in count_intent_sorted]

    print('intents:', intents)

    if 'No intent detected' in intents and remove_fallback == True:
        intents.remove('No intent detected')

    df_intents = df.loc[df['intent'].isin(intents)]


    return df_intents, count_intent, intents

def score_data(params, thre_augm = 0.9):
    """ Load and transform (embeddings) data to train score model """
    
    ### Prepare Data for training score_model
    # dialog_short = pd.read_csv(file)
    df_location = multiple_data_location(params, file_path='test_path', file_type='test_file') # will edit test_file later
    # df_location = multiple_data_location(params, file_path='data_path', file_type='data_file')
    dialog_short = read_multiple_files(df_location)

    scores_inv = np.array(dialog_short['feedback'])
    
    #revert scale such that: 0=bad; 1=good
    scores = np.where( scores_inv == 0, 1, 0) 
    reward = np.array(dialog_short['reward'])

    # CL used only for data augmentation
    scores_comb = list(0.5*np.add(scores, reward)) 
    
    scores = list(scores)
    
    utt= list(dialog_short['question'])
    resp = list(dialog_short['answer'])
    intents = list(dialog_short['intent'])

    # data augmentation for high CL answers
    intents_u = list(set(intents))

    ind = np.flatnonzero( np.array([np.array(scores_comb) > thre_augm ]))

    bad_intent = 'No intent detected'
    bad_resp = "Sorry, I can't help you with that. Please direct your question to the GS Local Support (GeneralServicesLocalSupportLausanne)"

    for ind0 in ind:

        utt.append(utt[ind0])
        intents.append(bad_intent)
        resp.append(bad_resp)
        scores.append(0)

        
    EMB = embeddings([], embed_par='tf')

    emb_utt = EMB.fit(utt)
    emb_resp = EMB.fit(resp)

    emb_s = EMB.embed_size

    #### Delete NaN
    ind = np.argwhere(np.isnan(emb_utt[:,0]))

    emb_utt = np.delete(emb_utt, ind, axis = 0)
    emb_resp = np.delete(emb_resp, ind, axis = 0)
    scores = np.delete(scores, ind, axis = 0 )

    m = np.shape(emb_utt)[0]

    concat = np.concatenate((emb_utt, emb_resp))

    x = concat.reshape([2, m, emb_s])
    y = scores.astype(np.float32)
    
    return x, y, emb_utt, emb_resp

class Simulator(object):

    def __init__(self,  conv_dict, multiply=4):
        """ initialize parent sample of conversation and random generator """
        
        self.conv_dict = conv_dict

        all_utt = list(conv_dict['question'])
        feedback = list(conv_dict['feedback'])
        
        ## define parent sample of conversations: self.utt such that extraction
        ## of question triggering resp  with negative feedback has higher probability
        ## probability ratio negative:positive defined by multiply
        wrongs = []
        good = []
        utt = []
        
        for i in range(len(all_utt)):
            if feedback[i] == 0:
                utt.append(all_utt[i])
            else:
                for k in range(multiply):
                    utt.append(all_utt[i])

        self.utt = utt             
        self.N = len(self.utt)
        
        print()
        print('---------------------------------------------------------------------------')
        print('Parent sample of conversations in Simulator contains %d questions'%self.N)

        ## init random generator
        seed = 45896
        random.seed(seed)

        self.iterator = conv_dict.iterrows()

        return

    def run_random(self):
        """ Returns random action """
        
        r = random.randint(0, self.N-1)       
        return self.utt[r]

    def sequential(self, reset=False):
        """ return action in sequential order """
        
        if reset:

            self.iterator = None
            self.iterator = self.conv_dict.iterrows()
            print('resetting...',self.iterator)

        try:
            return  next(self.iterator)[1]
        except StopIteration:
            return None
        



def state_to_dict(df):

    questions = list(df['question'])
    unique_questions = list(set(questions))
    print('# questions in final DF:', len(questions), len(list(set(questions))))

    count = 0
    map_index2state = {}
    map_state2index = {}

    for utt in unique_questions:

        map_index2state[count] = utt
        map_state2index[utt] = count

        count += 1

    return map_index2state, map_state2index

def get_mapping(NLU, df):
    """ return dictionary mapping set of states (questions) to indices and viceversa, and set of actions (answers) to indices and viceversa """
    
    map_index2action, map_action2index = NLU.actions_to_dict()
    map_index2state, map_state2index = state_to_dict(df)

    mapping = {}
    mapping['index2action'] = map_index2action
    mapping['action2index'] = map_action2index
    mapping['index2state'] = map_index2state
    mapping['state2index'] = map_state2index

    return mapping
