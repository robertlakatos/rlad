from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import Punctuation

class WordLevelVocab:
    
    def __init__(self, corpus, unk_token, vocab_size):
        self.name = "word_level"        
        self.corpus = corpus
        self.unk_token = unk_token
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(WordLevel(unk_token=self.unk_token))
        self.trainer = WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=[self.unk_token])
        self.preTokenizerSequence = Sequence([Whitespace(), Punctuation()])
        self.tokenizer.pre_tokenizer = self.preTokenizerSequence

    def get_vocab_size(self):
        return self.vocab_size + 1

    def train(self):
        self.tokenizer.train_from_iterator(self.corpus, self.trainer)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, data):
        return self.tokenizer.encode(data)
