import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

if __name__ == '__main__':


    # your code here
    args = sys.argv[1:]
    is_dynamic_rho = False
    reward_function_id = 0
    episodes = 1600

    if '-dynamic_rho' in args:
        is_dynamic_rho = True

    if '-reward_function' in args:
        reward_function_id = int(args[args.index('-reward_function') + 1])

    if '-episodes' in args:
        episodes = int(args[args.index('-episodes') + 1])

    if '-p_and_rho' in args:
        from modules.analysis import PAndRhoAnalysis
        path = args[args.index('-p_and_rho') + 1]
        PAndRhoAnalysis.plot(path)

    else:
        from utils import LogUtils
        LogUtils.info("MAIN", "START")
        import time
        try:
            start_time = time.time()
            if '-ra' in args:
                from modules.train import RiskAverseTrain
                train = RiskAverseTrain(
                    episodes=episodes,
                    is_dynamic_rho=is_dynamic_rho
                )
                train.start_train()
            elif '-dqn' in args:
                from modules.train import Train
                train = Train(
                    num_episode = episodes,
                    is_dynamic_rho = is_dynamic_rho,
                    reward_function_id = reward_function_id
                )
                train.start_train()
            elif '-ra_dqn' in args:
                from modules.train import RA_DQNTrain
                train = RA_DQNTrain(
                    episodes = episodes,
                    is_dynamic_rho = is_dynamic_rho
                )
                train.start_train()
            else:
                print("Invalid arguments!")
            end_time = time.time()
            LogUtils.info("MAIN", f"Time: {end_time - start_time}")
        except:
            print("Unexpected error:", sys.exc_info())
            LogUtils.delete_log()

        LogUtils.info("MAIN", "END")
        LogUtils.delete_log()


