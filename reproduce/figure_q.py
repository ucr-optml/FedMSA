import numpy as np
import os, sys
import yaml
import matplotlib.pyplot as plt

tau_list = [4, 8, 12]
q_list = [0.1, 0.3, 0.5, 1.0]
result_list_fednest = []
result_list_fedblo = []
colors = ["tab:blue", "tab:orange", "tab:green"]
for tau in tau_list:
    result_list_fednest = []
    result_list_fedblo = []

    for q in q_list:
        plt.cla()
        save_path = f"results/imbalance_blo/{tau}_{q}_frac_0.1.yaml"
        with open(save_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        result_list_fedblo.append((data["round"], data["test_acc"]))

        plt.plot(data["round"], data["test_acc"], label="FedMSA", linewidth=3)


        comm_round = data["round"][-1]
        print(comm_round)
        save_path = f"results/fednest/{tau}_{q}_frac_0.1.yaml"
        with open(save_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        # find the test acc at the nearest comm_round
        for i in range(len(data["round"])):
            if data["round"][i] >= comm_round:
                break
        result_list_fednest.append((data["round"], data["test_acc"]))

        # result_list_fednest.append(data["test_acc"][-1])

        plt.plot(data["round"], data["test_acc"], label="FedNest", linewidth=3)
        plt.xlim(0, 1000)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid()
        plt.legend(fontsize=16)
        plt.savefig(f"results/figs/q_{q}_tau_{tau}.pdf")
        plt.ylim(40, 95)
        plt.xlim(0, 2000)
        plt.savefig(f"results/figs/long_q_{q}_tau_{tau}.pdf")

    for q in q_list:
        plt.cla()
        save_path = f"results/imbalance_blo/{tau}_{q}_frac_0.1.yaml"
        with open(save_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        result_list_fedblo.append((data["round"], data["test_acc"]))

        plt.plot(data["test_acc"], label="FedMSA", linewidth=3)


        save_path = f"results/fednest/{tau}_{q}_frac_0.1.yaml"
        with open(save_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        # find the test acc at the nearest comm_round
        for i in range(len(data["round"])):
            if data["round"][i] >= comm_round:
                break
        result_list_fednest.append((data["round"], data["test_acc"]))


        plt.plot(data["test_acc"], label="FedNest", linewidth=3)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid()
        plt.legend(fontsize=16)
        plt.savefig(f"results/figs/epoch_q_{q}_tau_{tau}.pdf")
