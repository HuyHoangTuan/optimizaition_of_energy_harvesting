import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "./src")

if __name__ == '__main__':


    # your code here
    args = sys.argv[1:]
    is_dynamic_rho = False
    reward_function_id = 0
    episodes = 1600
    num_su = 1

    if '-dynamic_rho' in args:
        is_dynamic_rho = True

    if '-reward_function' in args:
        reward_function_id = int(args[args.index('-reward_function') + 1])

    if '-episodes' in args:
        episodes = int(args[args.index('-episodes') + 1])

    if '-su' in args:
        num_su = int(args[args.index('-su') + 1])

    if '-p_and_rho' in args:
        from src.modules.analysis import PAndRhoAnalysis
        path = args[args.index('-p_and_rho') + 1]
        PAndRhoAnalysis.plot(path)

    else:
        from src.utils import LogUtils
        import time
        try:
            LogUtils.info("MAIN", "START")
            start_time = time.time()
            if '-ra' in args:
                from src.modules.train import RiskAverseTrain
                train = RiskAverseTrain(
                    num_su=1,
                    episodes=episodes,
                    is_dynamic_rho=is_dynamic_rho
                )
                train.start_train()
            elif '-dqn' in args:
                from src.modules.train import Train
                train = Train(
                    num_episode = episodes,
                    is_dynamic_rho = is_dynamic_rho,
                    reward_function_id = reward_function_id
                )
                train.start_train()
            elif '-ra_dqn' in args:
                from src.modules.train import RA_DQNTrain
                train = RA_DQNTrain(
                    episodes = episodes,
                    is_dynamic_rho = is_dynamic_rho
                )
                train.start_train()
            else:
                print("Invalid arguments!")
            end_time = time.time()
            LogUtils.info("MAIN", f"Time: {end_time - start_time}")
            LogUtils.info("MAIN", "END")
            LogUtils.delete_log()
        except:
            print("Unexpected error:", sys.exc_info())
            LogUtils.delete_log()




