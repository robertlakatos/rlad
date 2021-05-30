import tensorflow as tf
from stable_baselines3.common.callbacks import BaseCallback
import os
from glob import glob

class DummyLogger:
    def __init__(self, log_dir):
        pass

    def write_metadata(self, run, key, value):
        print("run ", run, " ", key, ":", value)

class TensorboardLoggerSimple:
    def __init__(self, log_dir, run_name="run"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        folders = [folder_name for folder_name in os.listdir(log_dir) if folder_name.startswith(run_name)] # os.path.isdir(folder_name) and
        total_runs_so_far = len(folders)

        self.run_id = f"{run_name}_{total_runs_so_far + 1}"

        print("Logdir: ", self.run_id)

        self.tb_logger = tf.summary.create_file_writer(f"{log_dir}/{self.run_id}")

    def write_metadata(self, run, key, value):
        with self.tb_logger.as_default():
            tf.summary.scalar(key, value, step=run)

if __name__ == "__main__":
    logger = TensorboardLoggerSimple(log_dir="test_log")

    for i in range(10 + 1):
        logger.write_metadata(epoch=i, key="accuracy", value=i*10)