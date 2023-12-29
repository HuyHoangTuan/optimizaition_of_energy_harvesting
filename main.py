import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from utils import LogUtils

if __name__ == '__main__':
    LogUtils.info("MAIN", "START")

    # your code here
    from modules.train import Train

    try:
        import time
        start_time = time.time()
        train = Train()
        train.start_train()
        end_time = time.time()
        LogUtils.info("MAIN", f"Time: {end_time - start_time}")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        LogUtils.delete_log()

    LogUtils.info("MAIN", "END")
