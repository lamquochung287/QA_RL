import underthesea
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData

#JUST TRYING AND TESTING

class UndertheseaTokenizer(Component):
    # define the name of the component
    name = "underthesea_tokenizer"

    # define the required configuration keys
    required_components = []

    # define the default configuration
    defaults = {}

    # define the supported language list
    language_list = ["vi"]

    def __init__(self, component_config=None):
        super(UndertheseaTokenizer, self).__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        # tokenize each training example
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # tokenize the input message
        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # use underthesea to tokenize the text
        tokens = underthesea.word_tokenize(text)
        # convert the tokens to the rasa format
        return [Token(t, i) for i, t in enumerate(tokens)]
