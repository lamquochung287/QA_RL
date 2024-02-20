import pandas as pd
import numpy as np
from utils import multiple_data_location
from utils import read_multiple_files


class evaluate(object):
    """ evaluate RL performances on test set extracted from the full conversations and manually labeled """
    
    def __init__(self, params, map_state2index, map_index2action, file='data_chapter04_test_set.csv'):
        """ read test set """
        
        # DF = pd.read_csv(file)
        df_location = multiple_data_location(params, file_path='test_path', file_type='test_file')
        DF = read_multiple_files(df_location)
        
        self.utt_test = list(DF['question'])
        self.gold_resp = list(DF['gold_intent'])
        
        self.map_state2index = map_state2index
        self.map_index2action = map_index2action
        self.size_test = len(self.utt_test)

    def fit(self, agent):
        """ run evaluation"""
        
        success = 0
        num_utt = 0
        print('#### Start evaluationa #######')
        for i, question in enumerate(self.utt_test):

            if question in self.map_state2index:
                
                repr = get_representation(question, self.map_state2index)

                index_action = agent.run_policy(repr, None)
                
                if index_action in self.map_index2action:

                    num_utt += 1
                    action = self.map_index2action[index_action]
                    gold = self.gold_resp[i]
                    if action == gold:
                        success += 1
                    print('---------------------------------')
                    print('question:', question, i)
                    print('action:', action)
                    print('gold_intent:', gold)
                    print('success:', success)


        print('#### End of evaluation #######')
        success_rate =  float(success)/float(num_utt)       
        print('Success Rate:', success_rate, success, num_utt)

        return success_rate 

def get_representation(question, map_state2index):
    """ return one-hot representation of the given sentence """
    
    size = len(map_state2index)
    index = map_state2index[question]
    rep = np.zeros((1,size))
    rep[0, index] = 1.0

    return rep
