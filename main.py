import sys
from dotenv import load_dotenv
from src.utils import LogUtils
import matplotlib
import matplotlib.pyplot as plt
from src.utils import Parser
import torch
import math
import numpy as np

load_dotenv()
sys.path.insert(0, "./src")

if __name__ == '__main__':


    # your code here
    args = sys.argv[1:]
    is_dynamic_rho = False
    reward_function_id = 0
    episodes = 1600
    num_su = 1
    is_rnn = False
    is_hard_update = False

    if '-dynamic_rho' in args:
        is_dynamic_rho = True

    if '-reward_function' in args:
        reward_function_id = int(args[args.index('-reward_function') + 1])

    if '-episodes' in args:
        episodes = int(args[args.index('-episodes') + 1])

    if '-su' in args:
        num_su = int(args[args.index('-su') + 1])
    
    if '-rnn' in args:
        is_rnn = True

    if '-hard_update' in args:
        is_hard_update = True

    if '-plot' in args:
        plt.rcParams.update({'font.size': 24})
        plt.figure(num=1, figsize=(16, 9), dpi=100)

        ax = plt.subplot(1, 1, 1)
        name = ['dqn', 'rnn']
        SUs = [15, 15]
        is_bound = [0, 0, 0]
        datas = []
        bounds = None
        for i, su in enumerate(SUs):
            path = f'res/plot/{name[i]}_{su}.log'
            parser = Parser('dqn', path)
            reward, reward_t, rate, rate_t, loss, rho, rho_t, bound, bound_t, ta, ta_t, P = parser.get_data()
            datas.append(rho)
            if is_bound[i]:
                bounds = bound

        labels = ['Proposed DQN', 'Proposed DRQN']
        line_styles = ['-', '-', '-', '-', '-', '-']
        # markers = ['.', 's', 'x']
        # markers = ['.', 's', 'x']
        markers = ['s', 'x']
        # markers = ['.', 's', 'x']
        # markers = ['s', 's', 'x', 'x']
        # colors = ['peru', 'palegreen', 'crimson', 'navy']
        # colors = ['royalblue', 'darkorange', 'lime']
        colors = ['darkorange', 'lime']
        # colors = ['darkorange', 'burlywood', 'lime', 'seagreen']
        # colors = ['darkorange', 'burlywood', 'gold']
        # colors = ['lime', 'seagreen', 'aquamarine']
        # colors = ['lime', 'seagreen', 'aquamarine', 'seagreen', 'gold', 'aquamarine']
        f = 100
        # plt.title('rho')
        plt.ylabel('Average rho')
        plt.xlabel('Episode')
        for i, su in enumerate(SUs):
            mean_data = []
            for j in range(len(datas[i])):
                mean_data.append(torch.mean(torch.tensor(datas[i][:j][-f:], dtype=torch.float32)))
            plt.plot(torch.tensor(mean_data, dtype=torch.float32).numpy(), label=f'{labels[i]} {su} SUs', linestyle=line_styles[i], marker=markers[i], color=colors[i], linewidth=1.5, markevery=math.floor(len(datas[i])/32 ))

        if bounds is not None:
            mean_bound = []
            for i in range(len(bounds)):
                mean_bound.append(torch.mean(torch.tensor(bounds[:i][-f:], dtype=torch.float32)))
            plt.plot(torch.tensor(mean_bound, dtype=torch.float32).numpy(), label='Bound', linestyle='-', color='gray', linewidth=1.5)

        plt.legend(loc='best')
        ax.grid()
        plt.show()

    elif '-plot2' in args:
        plt.rcParams.update({'font.size': 24})

        def _plot():
            plt.figure(num=1, figsize=(16, 9), dpi=100)
            ax = plt.subplot(1, 1, 1)
            plt.ylabel('Rho/ C (Joule)/ P (Watt)')
            plt.xlabel('Time slot')
            x = [(i + 1) for i in range(len(data1s))]
            markers = [True if data2s[i] == 0 else False for i in range(len(data2s))]
            plt.bar(
                x,
                data3s,
                edgecolor='darkorange',
                color='darkorange',
                width=0.5,
                label="C"
            )
            plt.plot(
                x,
                data1s,
                marker='o',
                markersize=10,
                markevery=markers,
                label="Rho"
            )
            plt.plot(
                x,
                data4s,
                color='black',
                linestyle='--',
                label="P"
            )
            plt.xticks(x)
            plt.ylim(bottom=0, top=1.0)
            plt.legend(loc='upper right')
            ax.grid()
            plt.show()

        name = ['test_C']
        SUs = [1]
        data1s = None
        data2s = []
        data3s = []
        data4s = []
        x = []
        bounds = None
        for i, su in enumerate(SUs):
            path = f'res/plot/{name[i]}_{su}.log'
            parser = Parser('dqn', path)
            reward, reward_t, rate, rate_t, loss, rho, rho_t, bound, bound_t, ta, ta_t, P = parser.get_data()
            C, P_t = parser.get_C()
            for i_episode in range(episodes):
                _rate = reward[i_episode]
                _rate_t = rate_t[i_episode]
                _ta_t = ta_t[i_episode]
                _rho_t = rho_t[i_episode]
                _C_t = C[i_episode]
                _P = P_t[i_episode]

                if data1s is None and np.sum(np.array(_ta_t)) >= 10 and np.sum(np.array(_ta_t)) <= 17 and _rate >= 40.0:
                    data1s = _rho_t
                    data2s = _ta_t
                    data3s = _C_t
                    data4s = _P
                    print(
                        f'{i_episode}\n'
                        f'{[f"{num:.2f}" for num in _C_t]}\n'
                        f'{[f"{num}" for num in _ta_t]}\n'
                        f'{[f"{num:.2f}" for num in _rho_t]}\n'
                        f'{[f"{num:.2f}" for num in _rate_t]}\n'
                        f'{_rate}\n'
                        f'----------------------'
                    )

                    _plot()

        # labels = ['Proposed DRQN', 'Proposed DRQN', 'Proposed DRQN']
        # if data1s is not None:
        #     _plot()
    else:
        LogUtils.info("MAIN", "START")
        import time
        try:
            start_time = time.time()
            if '-ra' in args:
                from src.modules.train import RiskAverseTrain
                train = RiskAverseTrain(
                    episodes=episodes,
                    is_dynamic_rho=is_dynamic_rho
                )
                train.start_train()
            elif '-dqn' in args:
                from src.modules.train import Train
                train = Train(
                    num_su=num_su,
                    num_episode = episodes,
                    is_dynamic_rho = is_dynamic_rho,
                    reward_function_id = reward_function_id,
                    is_rnn=is_rnn,
                    is_hard_update=is_hard_update
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


