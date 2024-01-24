from typing import Text, List
from rasa.nlu.tokenizers import Tokenizer, Token
from underthesea import UndertheseaTokenizer
import underthesea

#JUST TRYING AND TESTING

class UndertheseaTokenizer(Tokenizer):
    def tokenize(self, text: Text) -> List[Token]:
        # Use Underthesea for tokenization
        underthesea_tokens = underthesea.word_tokenize(text)
        
        # Convert Underthesea tokens to Rasa Token objects
        rasa_tokens = [Token(token, start) for (token, start, end) in self._convert_tokens(underthesea_tokens, text)]

        return rasa_tokens

    def _convert_tokens(self, tokens: List[str], text: Text) -> List[Tuple[Text, int, int]]:
        """
        Converts Underthesea tokens to Rasa Token objects.
        """
        offsets = []
        start = 0
        for token in tokens:
            start = text.find(token, start)
            end = start + len(token)
            offsets.append((token, start, end))
            start = end
        return offsets
    
