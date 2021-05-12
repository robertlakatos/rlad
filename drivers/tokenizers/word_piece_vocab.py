from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import Punctuation

class WordPieceVocab:
    
    def __init__(self, corpus, unk_token, vocab_size):
        self.name = "word_piece"     
        self.corpus = corpus
        self.unk_token = unk_token
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(WordPiece(unk_token=self.unk_token))
        self.trainer = WordPieceTrainer(vocab_size=self.vocab_size, special_tokens=[self.unk_token])
        self.preTokenizerSequence = Sequence([Whitespace(), Punctuation()])
        self.tokenizer.pre_tokenizer = self.preTokenizerSequence

    def get_vocab_size(self):
        return self.vocab_size

    def train(self):
        self.tokenizer.train_from_iterator(self.corpus, self.trainer)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, data):
        return self.tokenizer.encode(data)
