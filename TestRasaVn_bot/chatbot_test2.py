# import pandas as pd
import numpy as np
from utils import multiple_data_location
from utils import read_multiple_files
from utils import parameters
from deep_dialog.qlearning import DQN
from deep_dialog.nlu.rasa import rasa
from deep_dialog.nlu.rasa import Interpreter
import pickle
import copy
import random
from pathlib import Path

def get_mapping(training_dict, no_intent, df, map_action2answer):
    """ return dictionary mapping set of states (questions) to indices and viceversa, and set of actions (answers) to indices and viceversa """
    
    map_index2action, map_action2index = actions_to_dict(training_dict, no_intent)
    map_index2state, map_state2index = state_to_dict(df)

    return {
        'index2action': map_index2action,
        'action2index': map_action2index,
        'index2state': map_index2state,
        'state2index': map_state2index,
        'action2answer': map_action2answer,
    }

def state_to_dict(df):

    questions = list(df['question'])
    unique_questions = list(set(questions))
    # print('# questions in final DF:', len(questions), len(list(set(questions))))

    map_index2state = {}
    map_state2index = {}

    for count, utt in enumerate(unique_questions):

        map_index2state[count] = utt
        map_state2index[utt] = count

    return map_index2state, map_state2index


def actions_to_dict(training_dict, no_intent):

    print('Creating action to index mapping')

    count = 0
    map_index2action = {}
    map_action2index = {}

    data = training_dict
    example_list=[]
    for i in range(len(data)):
        example_list.extend(data[i]["rasa_nlu_data"]["common_examples"])

    intent_list = [item['intent'] for item in example_list]
    u_intent_list = list(set(intent_list))

    if no_intent not in u_intent_list:
        u_intent_list.append(no_intent)

    for count, action in enumerate(u_intent_list):
        #in rasa action == intent

        map_action2index[action] = count
        map_index2action[count] = action
        
    return map_index2action, map_action2index



###############################################################
# def get_representation(question, map_state2index):
#     """ return one-hot representation of the given sentence """
    
#     size = len(map_state2index)
#     index_question = map_state2index[question]
#     rep = np.zeros((1,size))
#     rep[0, index_question] = 1.0

#     return rep
def get_representation(question, map_state2index):
    """
    Return a one-hot representation of the given sentence.

    Args:
        question (str): The input sentence.

    Returns:
        np.ndarray: The one-hot representation of the sentence.
    """
    # Extract unique words from the question and create a mapping
    unique_words = sorted(set(question.split()))
    map_word2index = {word: i for i, word in enumerate(unique_words)}

    # Determine the size of the representation based on the number of unique words
    # size = len(unique_words)
    size = len(map_state2index) 
    
    # Create the one-hot representation
    representation = np.zeros((1, size))
    for word in question.split():
        representation[0, map_word2index[word]] = 1.0

    return representation

def load_trained_DQN(path):
    """ Load trained model from pickle file """
    
    trained_file = pickle.load(open(path, 'rb'))
    model = trained_file['model']

    print(f'DQN loaded from file: {path}')
    # print()
    
    return model
    
###################################
def softmax(x, tau=1.0):
    """Compute softmax values for each value in x."""
    e_x = np.exp((x - np.max(x)) / tau)
    return e_x / e_x.sum()

def softmax_action_selection(action_values, temperature=1.0):
    """Select action based on softmax action selection."""
    probabilities = softmax(action_values / temperature)
    action = np.random.choice(len(probabilities), p=probabilities)
    return action

def update_epsilon(min_epsilon, epsilon, epsilon_decay):
        """Update the exploration rate (epsilon) based on the decay schedule."""
        return max(min_epsilon, epsilon * epsilon_decay)
        
def run_policy_DQN(dqn, representation, cfgs, num_actions, return_q=False):
    """Epsilon-greedy policy with a decaying exploration rate."""
    gamma = cfgs['gamma']
    epsilon_initial = cfgs['epsilon_initial']
    min_epsilon = cfgs['min_epsilon']
    epsilon_decay = cfgs['epsilon_decay']
    
    epsilon = update_epsilon(min_epsilon, epsilon_initial, epsilon_decay)
    
    if np.random.rand() < epsilon:
        # Exploration: choose a random action
        action = np.random.randint(num_actions)
    else:
        # Exploitation: choose the action with the highest Q-value
        action = dqn.predict(representation, {'gamma': gamma}, predict_model=True)
        
    # if return_q:
    #     # Optionally return Q-values for the chosen action
    #     q_value = dqn.predict(representation, {'gamma': gamma}, predict_model=True, return_q=True)
    #     return action, q_value
    # else:
    return action


#######################################
# def run_policy_DQN(dqn, representation, return_q=False):
#     """ epsilon-greedy policy """

#     return dqn.predict(representation, {'gamma': 0.9}, predict_model=True, return_q=return_q)
            
def rule_policy_NLU(nlu, user_question):

    return nlu.predict(user_question)

def predict_NLU(nlu, user_question, mapping, cl_threshold):
    """ predict action (with confidence) given a state (utterance) """
    
    result = nlu.interpreter.parse(user_question)
    intent = result['intent']['name']
    confidence = result['intent']["confidence"]

    if confidence < cl_threshold:
        intent = 'No intent detected'
        confidence = 1.-confidence

    map_action2answer = mapping['action2answer']
    answers = [map_action2answer.loc[map_action2answer['intent'] == intent]['answer'].values[0]]
        
    return answers, intent
####################################################

def load_data(params):    
    
    df_location = multiple_data_location(params, file_path='data_path', file_type='data_file')
    df = read_multiple_files(df_location)
    
    df.dropna(subset=['reward'], how='all', inplace = True)
    df.dropna(subset=['question'], how='all', inplace = True)
    df.dropna(subset=['answer'], how='all', inplace = True)
    
    df_intents = get_intents(df)
    
    # remove feedback intent, as it does not have answer
    df_intents = df_intents[df_intents['intent'] != 'Feedback']
    
    return df_intents

def get_intents(df, remove_fallback=True):
    """ This function selects the max_num_intents intents more frequently triggered by the users and return dataframe with these intents only"""

    # df_feedback = get_df_feedback(df)

    intents = list(df['intent'])
    unique_intents = list(set(intents))

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

    if 'No intent detected' in intents and remove_fallback == True:
        intents.remove('No intent detected')

    df_intents = df.loc[df['intent'].isin(intents)]


    return df_intents

#######################################################    

def set_mapping(params):
    
    df = load_data(params)
    no_intent = "No intent detected"
    
    df_location = multiple_data_location(params, file_path='training_path', file_type='training_file')
    training_dict = read_multiple_files(df_location, 'json')
    
    df_location = multiple_data_location(params, file_path='intent_response_path',file_type='intent_response_file')
    map_action2answer = read_multiple_files(df_location)
    
    return get_mapping(training_dict, no_intent, df, map_action2answer)

# def chatbot_response(user_question, mapping, dqn):  # sourcery skip: avoid-builtin-shadow

#     # print('state2index: ',mapping['state2index'])
#     map_state2index = mapping['state2index']
#     if user_question in map_state2index:
        
#         repr = get_representation(user_question, map_state2index)

#         index_action = run_policy_DQN(dqn, repr)

#         map_index2action = mapping['index2action']
#         map_action2answer = mapping['action2answer']
#         # print('index2action: ',map_index2action)
#         # print('action2answer: ',map_action2answer)

#         if index_action in map_index2action:
#             answers = [map_action2answer.loc[map_action2answer['intent'] == map_index2action[index_action]]['answer'].values[0]]
#             print('-------answers: ', answers)
            
#             return map_index2action[index_action]
        
#     return "I have no idea"

def chatbot_response(user_question, mapping, dqn, cfgs, num_actions):  # sourcery skip: avoid-builtin-shadow
    
    map_state2index = mapping['state2index']
    repr = get_representation(user_question, map_state2index)

    index_action = run_policy_DQN(dqn, repr, cfgs, num_actions)

    map_index2action = mapping['index2action']
    map_action2answer = mapping['action2answer']

    if index_action in map_index2action:
        answers = [map_action2answer.loc[map_action2answer['intent'] == map_index2action[index_action]]['answer'].values[0]]
        # print('-------answers: ', answers)

        return answers, map_index2action[index_action]

    return "I have no idea", index_action

####################################################
def read_dqn_model():
    file_paths = [f'./models/20240219/agt_{i}.p' for i in range(9) if Path(f'./models/20240219/agt_{i}.p').is_file()]
    model_params_list = [load_trained_DQN(file_path) for file_path in file_paths]
    merged_params = {
        key: np.mean(np.array([model_params[key] for model_params in model_params_list]), axis=0)
        for key in model_params_list[0].keys()
    }
    return merged_params

def set_nlu_model(cfgs):
    rasa_NLU = rasa(None, None, isTest=True)
    rasa_NLU.cl_threshold = cfgs['cl_threshold']
    rasa_NLU.no_intent = cfgs['no_intent']
    rasa_NLU.interpreter = Interpreter.load(cfgs['rasa_model_path'])
    
    return rasa_NLU
    
def config():
    cfgs = {
        'gamma': 0.9,
        'epsilon_initial': 1.0,
        'min_epsilon': 0.01,
        'epsilon_decay': 0.5,
        'cl_threshold': 0.15,
        'no_intent': "Hello, have a nice day!",
        'rasa_model_path': './projects/default/nlu_20240218-113234_checkpoint_13h',
    }
    
    return cfgs
    
def chat_bot():
    
    cfgs = config()
    params = parameters()
    mapping = set_mapping(params)
    num_actions = len(mapping['index2action'])
    
    dqn = DQN(None, None, None, True)
    dqn.model = copy.deepcopy(read_dqn_model())

    rasa_NLU = set_nlu_model(cfgs)

    print("Chatbot: Hi! How can I help you today?")
    while True:
        user_question = input("You: ")
        if user_question.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response, intent_DQN  = chatbot_response(user_question, mapping, dqn, cfgs, num_actions)
        response2, intent_predict_NLU = predict_NLU(rasa_NLU, user_question, mapping, cfgs)
        response3, intent_rule_NLU, confi = rule_policy_NLU(rasa_NLU, user_question)
        print(f"Chatbot_DQN: {response} /// {intent_DQN}")
        print()
        print(f"Chatbot_NLU: {response2} /// {intent_predict_NLU}")
        print()
        print(f"Policy_NLU_base:{response3} /// {intent_rule_NLU} /// {confi}")
        print("====================================================")

chat_bot()
