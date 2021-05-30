import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
import pandas as pd
import math
from .util_tensorboard import TensorboardLoggerSimple, DummyLogger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import os
import time
import torch
from tensorflow.keras.layers import Input


def get_reward(model,
               #done,
               X_train,
               y_train,
               X_val,
               y_val,
               preprocess_fn_X,
               monitor,
               lambda_=1,
               patience=3,
               max_epochs=1000):
    # Accuracy + Validation Accuracy - lambda * (Accuracy - Validation Accuracy)
    # Ötlet:
    #   - minél jobb a pontosság a tanító adathalmazon -> annál nagyobb jutalom
    #   - minél jobb a pontosság a validációs adathalmazon -> annál nagyobb jutalom
    #   - minél nagyobb a különbség a tanító adathalmazonmért pontosság és a validációson mért között -> annál kevesebb jutalom (büntetés)
    #   - lambda_ a büntetőtag (szerintem az 1 maradhat neki, de majd TODO megnézni)

    #if not done:
    #    return 0

    X_train_preprocessed = preprocess_fn_X(X_train)
    X_val_preprocessed = preprocess_fn_X(X_val)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience)

    history = model.fit(X_train_preprocessed,
                        y_train,
                        epochs=max_epochs,
                        batch_size=32,
                        validation_data=(X_val_preprocessed, y_val),
                        verbose=1,
                        callbacks=early_stopping_callback)

    acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]

    print("ACC: ", acc, "VAL_ACC: ", val_acc)

    # TODO: Más metrikák figyelembevétele?
    return (acc + val_acc) / 2 - lambda_ * math.fabs(
        acc - val_acc
    )  # TODO: VOCAB méretének büntetése? kell-e esetleg ha úgyis van val?


class VocabEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    run_idx = 0

    def __init__(self,
                 X_train,
                 y_train,
                 X_val,
                 y_val,
                 possible_words,
                 input_length,
                 n_classes,
                 model,
                 monitor="val_loss",
                 patience=3,
                 max_epochs=1000,
                 logger=DummyLogger(log_dir=None)):
        super(VocabEnv, self).__init__()

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.state = None
        self.vocab_built = None
        self.possible_words = possible_words
        self.input_length = input_length
        self.n_classes = n_classes

        self.monitor = monitor
        self.patience = patience
        self.max_epochs = max_epochs

        self.n_actions = len(possible_words) + 1  # +1 for the action meaning stop the search
        self.terminate_idx = self.n_actions - 1
        self.model = model
        self.model_built = None
        self.logger = logger

        VocabEnv.run_idx = 0
        self.total_reward = 0

        self.action_space = spaces.Discrete(self.n_actions)

        self.observation_space = spaces.Box(low=0,
                                            high=self.n_actions,
                                            shape=(self.n_actions, ),
                                            dtype=np.uint8)

    def _tokenize(self, data):
        tokenizer = BertWordPieceTokenizer(self.vocab_built, lowercase=True)

        return data.apply(lambda x: tokenizer.encode(x).ids)

    def _preprocess_input(self, data):
        tokenized = self._tokenize(data)

        return tf.keras.preprocessing.sequence.pad_sequences(
            tokenized, maxlen=self.input_length)

    # TODO: kicserélni
    def _build_model(self):
        print("Creating model with Vocab size ", len(self.vocab_built))
        print(self.vocab_built)
        print("-------------------------")

        self.model._create_model(len(self.vocab_built))

        return self.model.model # TODO: get_model

    def _add_to_vocab(self, idx):
        self.state[idx] = 1

        word = self.possible_words[idx]
        token = len(self.vocab_built)

        self.vocab_built[word] = token

    def step(self, action):
        new_token_index = action

        new_state = None
        reward = None
        done = False
        info = dict()

        # If the given token is already in the vocab, the current episode terminates and we penalize the agent
        if self.state[new_token_index] != 0:
            reward = -1
            done = True
        # If the agent chose to stop the search, we calculate the final reward
        else:
            # TODO: ha még túl kicsi a vocab mérete, akkor büntessünk (min_vocab_size)
            if new_token_index == self.terminate_idx:
                done = True

                self.model_built = self._build_model()
                reward = get_reward(model=self.model_built,
                                    #done=done,
                                    X_train=self.X_train,
                                    y_train=self.y_train,
                                    X_val=self.X_val,
                                    y_val=self.y_val,
                                    monitor=self.monitor,
                                    patience=self.patience,
                                    max_epochs=self.max_epochs,
                                    preprocess_fn_X=self._preprocess_input)
            else:
                done = False
                reward = 0

                # If the token is not stored in the vocab yet, we add it to the list of words
                self._add_to_vocab(new_token_index)

        new_state = self.state

        self.total_reward += reward

        # TODO:
        # maybe_ep_info = info.get("episode")
        # maybe_is_success = info.get("is_success")

        return (new_state, reward, done, info)

    def reset(self):
        self.logger.write_metadata(run=VocabEnv.run_idx,
                                   key="total_reward",
                                   value=self.total_reward)

        self.state = np.zeros(self.n_actions, dtype=np.uint8)  # Empty vocab
        self.vocab_built = {"[UNK]": 0, "[SEP]": 1, "[CLS]": 2}

        VocabEnv.run_idx += 1
        self.total_reward = 0

        return self.state

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class VocabSearch:
    def __init__(self,
                 X_train,
                 y_train,
                 possible_words,
                 n_classes,
                 model,
                 monitor="val_loss",
                 patience=3,
                 max_epochs=1000,
                 X_val=None,
                 y_val=None,
                 val_fraction=0.2,
                 input_length=128,
                 logger=DummyLogger(None)):
        if X_val is None:
            data_length = len(X_train)

            val_indices = np.random.choice(data_length,
                                           int(data_length * val_fraction))

            self.X_val = X_train.iloc[val_indices]
            self.y_val = y_train.iloc[val_indices]
            self.X_train = X_train.iloc[~val_indices]
            self.y_train = y_train.iloc[~val_indices]

            self.possible_words = possible_words
            self.n_classes = n_classes
            self.input_length = input_length
            self.logger = logger
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val

        self.model = model
        self.monitor = monitor
        self.patience = patience
        self.max_epochs = max_epochs

    def search(self, n_envs=1, single_thread=False):
        if single_thread:
            wrapper = DummyVecEnv
        else:
            wrapper = SubprocVecEnv

        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])

        def make_env(rank, seed=0):
            """
            Utility function for multiprocessed env.
            
            :param env_id: (str) the environment ID
            :param seed: (int) the inital seed for RNG
            :param rank: (int) index of the subprocess
            """
            def _init():
                env = VocabEnv(X_train=self.X_train,
                               y_train=self.y_train,
                               X_val=self.X_val,
                               y_val=self.y_val,
                               possible_words=self.possible_words,
                               input_length=self.input_length,
                               n_classes=4,
                               model=self.model,
                               logger=self.logger,
                               monitor=self.monitor,
                               patience=self.patience,
                               max_epochs=self.max_epochs
                               )
                env.seed(seed + rank)
                return env

            return _init

        model = PPO(
            "MlpPolicy",
            wrapper([make_env(i) for i in range(n_envs)]),
            learning_rate=3e-4,
            batch_size=512,
            n_steps=32768,
            n_epochs=5,
            verbose=0,
            device="cpu",
            policy_kwargs=policy_kwargs)

        model.learn(total_timesteps=100000)

        if not os.path.exists("models"):
            os.makedirs("models")

        model.save(f"models/{int(time.time())}")

        self.model_rl = model

        return self.model_rl

    def get_vocab(self):
        # TODO:
        # self.model_rl
        pass
        #return env.vocab_built

    # TODO
    def encode(self, word):
        vocab = self.get_vocab()
        default_value = vocab["UNK"]

        return vocab.get(word, default_value)


if __name__ == "__main__":
    df_train = pd.DataFrame({"text": ["apple", "banana"], "label": [0, 1]})
    X_train = df_train["text"]
    y_train = df_train["label"]

    df_val = pd.DataFrame({"text": ["pineapple", "orange"], "label": [1, 1]})
    X_val = df_val["text"]
    y_val = df_val["label"]

    VOCAB = ["apple", "banana", "pineapple", "orange"]

    env = VocabEnv(X_train,
                   y_train,
                   X_val,
                   y_val,
                   possible_words=VOCAB,
                   input_length=128,
                   n_classes=2)

    trajectories = [[0, 1, 2, 3, 4], [0, 1, 4]]
    episodes = len(trajectories)

    for i in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        idx = 0
        trajectory = trajectories[i]

        print("Start state: ", state)
        print("Simulated trajectory: ", trajectory)
        print("Simulation started...")

        while not done:
            new_state, reward, done, _ = env.step(trajectory[idx])

            print("New state: ", new_state)

            state = new_state
            total_reward += reward

            idx += 1

        print("Total reward: ", total_reward)
        print("Vocab built: ", env.vocab_built)
