from drivers.loaders.imdb import IMDB

from drivers.tokenizers.word_piece_vocab import WordPieceVocab
from drivers.tokenizers.word_level_vocab import WordLevelVocab
from drivers.tokenizers.unigram_vocab import UnigramVocab
from drivers.tokenizers.bpe_vocab import BPEVocab   
from drivers.rl.util_tensorboard import TensorboardLoggerSimple, DummyLogger
from drivers.rl.vocab_search import VocabEnv, VocabSearch
import os
from drivers.models.simple import Simple
import signal
import sys

VOCAB_SIZE = 1000
UNK_TOKEN = "[UNK]"

PATH_VOCABS = "vocabs/"
vocab_search = None

def ctrl_c(sig, frame):
    print("Terminating.")
    
    vocab_search.save_rl_model(name="terminated")

    sys.exit(0)

signal.signal(signal.SIGINT, ctrl_c)

def main():
    global vocab_search
    db = IMDB("data")

    vocab = WordLevelVocab(db.get_train()["text"].values, UNK_TOKEN, VOCAB_SIZE)

    file_name_vocab = PATH_VOCABS + vocab.name + "_" + db.name + ".json"

    print(file_name_vocab)

    if os.path.isfile(file_name_vocab) == False:
        vocab.train()
        print("TRAINED:", db.name)
        
        vocab.save(file_name_vocab)
        print("SAVED:", db.name)
    else:
        vocab.load(file_name_vocab)
        print("LOADED:", db.name)
        
        

    X_train = db.get_train()["text"]
    y_train = db.get_train()["label"]
    X_test = db.get_train()["text"]
    y_test = db.get_test()["label"]
    n_classes = len(y_train.unique())

    #env = VocabEnv(X_train, y_train, X_val, y_val, possible_words=vocab.tokenizer.get_vocab(), input_length=128, n_classes=n_classes)

    simple_name = "Simple_" + db.name + "_" + vocab.name

    model = Simple(input_length=128, output_size=db.get_labels(),
                    repeate=1,
                    name=simple_name)

    #vocab_search = VocabSearch(X_train.values, y_train.values, vocab.tokenizer.get_vocab(), n_classes, input_length=128, logger=DummyLogger(log_dir="tb_logs"))
    vocab_search = VocabSearch(X_train, y_train, {v: k for k, v in vocab.tokenizer.get_vocab().items()}, n_classes, model=model, input_length=128, logger=DummyLogger(log_dir=""), min_vocab_size=2)


    vocab_search.search(n_envs=10, single_thread=False)

if __name__ == "__main__":
    main()
