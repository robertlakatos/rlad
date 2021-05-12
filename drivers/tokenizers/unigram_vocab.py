from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import Punctuation
 
class UnigramVocab:
    
    def __init__(self, corpus, unk_token, vocab_size):
        self.name = "unigram"   
        self.corpus = corpus
        self.unk_token = unk_token
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(Unigram())
        self.trainer = UnigramTrainer(vocab_size=self.vocab_size, 
                                      unk_token=self.unk_token, 
                                      special_tokens=[self.unk_token])
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
