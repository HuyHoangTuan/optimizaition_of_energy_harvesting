import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def plot(file_in):
    with open(file_in, 'r') as f:
        lines = f.readlines()
        count = 0

        ks = []
        Ps = []
        Rhos = []
        for line in lines:
            prefix = '[TRAIN_EPISODE]'
            if line.find(prefix) != -1:
                count += 1
                start = line.find(':', line.find('action_value')) + 1
                k = int(float(line[start:line.find(',', start)].strip()))

                start = line.find(',', start) + 1
                P = float(line[start:line.find(',', start)].strip())

                start = line.find(',', start) + 1
                Rho = float(line[start:line.find(',', start)].strip())

                ks.append(k)
                Ps.append(P)
                Rhos.append(Rho)
        plt.figure(num = 1, figsize = (16, 9), dpi = 120)

        # Plot k
        # ---------------------------------------------------------------------
        plt.subplot(2, 2, 1)
        plt.bar(['0', '1'], [ks.count(0), ks.count(1)], edgecolor='black')
        for i in range(2):
            plt.text(i, ks.count(i), str(ks.count(i)), ha = 'center', va = 'bottom')

        plt.ylabel('Number')
        plt.title('The chosen k')
        # ---------------------------------------------------------------------

        # Plot Ps
        # ---------------------------------------------------------------------
        plt.subplot(2, 2, 2)
        plt.hist(Ps, bins = 100, edgecolor = 'black')

        plt.ylabel('Number')
        plt.title('The chosen P')
        # ---------------------------------------------------------------------

        # Plot Rho
        # ---------------------------------------------------------------------
        plt.subplot(2, 2, 3)
        plt.hist(Rhos, bins = 10, edgecolor = 'black')

        plt.ylabel('Number')
        plt.title('The chosen Rho')
        # ---------------------------------------------------------------------

        plt.tight_layout()
        plt.show()