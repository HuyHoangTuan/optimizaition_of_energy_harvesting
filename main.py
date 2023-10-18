import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from utils import LogUtils
from modules.train import Train

if __name__ == '__main__':
    LogUtils.info("MAIN", "START")

    # your code here
    train = Train()
    train.start_train()


    LogUtils.info("MAIN", "END")
