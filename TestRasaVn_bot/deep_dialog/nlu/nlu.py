
# from deep_dialog.nlu.watson import  watson
from deep_dialog.nlu.rasa import  rasa

def set_nlu(params, nlu_type, intents, **kargs):

        
        print('-----------  NLU in use is %s -----------'%nlu_type)
        
        # if nlu_type == 'watson':
        #     nlu = watson(params, intents, **kargs)
        if nlu_type == 'rasa':
            nlu = rasa(params, intents, **kargs)

        return nlu
