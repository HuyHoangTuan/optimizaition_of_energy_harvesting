import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

if __name__ == '__main__':


    # your code here
    args = sys.argv[1:]

    if '-p_and_rho' in args:
        from modules.analysis import PAndRhoAnalysis
        path = args[args.index('-p_and_rho') + 1]
        PAndRhoAnalysis.plot(path)
    else:
        from utils import LogUtils
        from modules.train import Train

        LogUtils.info("MAIN", "START")

        try:
            is_dynamic_rho = False
            reward_function_id = 0
            episodes = 1600

            if '-dynamic_rho' in args:
                is_dynamic_rho = True

            if '-reward_function' in args:
                reward_function_id = int(args[args.index('-reward_function') + 1])

            if '-episodes' in args:
                episodes = int(args[args.index('-episodes') + 1])

            import time
            start_time = time.time()
            train = Train(
                num_episode = episodes,
                is_dynamic_rho = is_dynamic_rho,
                reward_function_id = reward_function_id
            )
            train.start_train()
            end_time = time.time()
            LogUtils.info("MAIN", f"Time: {end_time - start_time}")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            LogUtils.delete_log()

        LogUtils.info("MAIN", "END")


