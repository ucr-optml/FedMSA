import numpy as np
import os, sys
import yaml
import matplotlib.pyplot as plt

tau_list = [4, 8, 12]
q_list = [0.1, 0.3, 0.5, 1.0]
result_list_fednest = []
result_list_fedblo = []

epoch = 100
for q in q_list:
    result_list_fednest = []
    result_list_fedblo = []
    plt.cla()
    for tau in tau_list:
        save_path = f"results/imbalance_blo/{tau}_{q}_frac_0.1.yaml"
        with open(save_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        result_list_fedblo.append(data["test_acc"][-1])

        # comm_round = data["round"][-1] * 10
        # # print(comm_round)
        i = len(data["test_acc"])

        save_path = f"results/fednest/{tau}_{q}_frac_0.1.yaml"
        with open(save_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        # # find the test acc at the nearest comm_round
        # for i in range(len(data["round"])):
        #     if data["round"][i] >= comm_round:
        #         break

        result_list_fednest.append(data["test_acc"][i])

    bar_width = 0.35
    x = np.arange(len(tau_list))
    plt.bar(x, result_list_fedblo, width=bar_width, label="FedMSA")
    plt.bar(
        x + bar_width, result_list_fednest, width=bar_width, label="FedNest", alpha=0.7
    )
    plt.xticks(x + bar_width / 2, tau_list, fontsize=16)

    plt.ylim(85, 94)
    yticks = [85, 86, 88, 90, 92, 94]
    yticklabels = ["~", "86", "88", "90", "92", "94"]
    plt.yticks(yticks, yticklabels, fontsize=16)

    plt.grid()
    plt.legend(fontsize=16)
    plt.savefig(f"results/figs/tau_{q}.pdf")
