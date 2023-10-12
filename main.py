import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from utils import LogUtils
from modules.environment import Environment

if __name__ == '__main__':
    LogUtils.info("MAIN", "START")

    # your code here
    env = Environment()
    State = env.reset()
    print(env.actions_space)

    LogUtils.info("MAIN", "END")
