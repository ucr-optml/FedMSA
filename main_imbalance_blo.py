#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import yaml
import time
from core.test import test_img
from utils.Fed import FedAvg, FedAvgGradient
from models.SvrgUpdate import LocalUpdate
from utils.options import args_parser
from utils.dataset import load_data
from models.ModelBuilder import build_model
from core.ClientManage_blo import ClientManage
from utils.my_logging import Logger
from core.function import assign_hyper_gradient, assign_weight_value
from torch.optim import SGD
import torch

import numpy as np
import copy

import gc

start_time = int(time.time())

if __name__ == "__main__":
    # parse args
    args = args_parser()
    (
        dataset_train,
        dataset_test,
        dict_users,
        args.img_size,
        dataset_train_real,
    ) = load_data(args)
    net_glob = build_model(args)

    # copy weights
    w_glob = net_glob.state_dict()
    if args.output == None:
        logs = Logger(f"./save/fed{args.optim}_{args.dataset}\
_{args.model}_{args.epochs}_C{args.frac}_iid{args.iid}_\
{args.lr}_blo{not args.no_blo}_\
IE{args.inner_ep}_N{args.neumann}_HLR{args.hlr}_{args.hvp_method}_{start_time}.yaml"
                      )
    else:
        logs = Logger(args.output)

    mu = 0.01**(1 / 9)
    probability = np.array([mu**-i for i in range(0, 10)])
    wy = probability / np.linalg.norm(probability)
    ly = np.log(1.0 / probability)
    hyper_param = {
        "dy": torch.zeros(args.num_classes,
                          requires_grad=True,
                          device=args.device),
        "ly": torch.zeros(args.num_classes,
                          requires_grad=True,
                          device=args.device),
        "wy": torch.tensor(wy, device=args.device, dtype=torch.float32),
    }

    comm_round = 0
    hyper_optimizer = SGD([hyper_param[k] for k in hyper_param], lr=1)
    h_last, q_last, p_last, s_last, v_last = None, None, None, None, None

    # 5 epoch of warmup
    for iter in range(5):
        m = max(int(args.frac * args.num_users), 1)
        client_idx = np.random.choice(range(args.num_users), m, replace=False)
        client_manage = ClientManage(args, net_glob, client_idx, dataset_train,
                                     dict_users, hyper_param)
        w_glob, loss_avg = client_manage.fed_in()
        if args.optim == "svrg":
            comm_round += 2
        else:
            comm_round += 1
        net_glob.load_state_dict(w_glob)
    print("Warmup done")
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train_real, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print(
        "Test acc/loss: {:.2f} {:.2f}".format(acc_test, loss_test),
        "Train acc/loss: {:.2f} {:.2f}".format(acc_train, loss_train),
        f"Comm round: {comm_round}",
    )

    # FedBLO
    for iter in range(args.epochs):
        # monitor memory usage
        r_init = torch.cuda.memory_allocated()

        m = max(int(args.frac * args.num_users), 1)
        if iter == 0:
            rho = 1.0
        else:
            rho = args.momentum_rho

        # compute global parameters
        client_idx = list(range(args.num_users))
        client_manage = ClientManage(args, net_glob, client_idx, dataset_train,
                                     dict_users, hyper_param)
        (
            h_glob,
            q_glob,
            p_locals,
            s_locals,
            v_glob,
            comm_round_func,
        ) = client_manage.fed_globhq(rho, h_last, q_last, p_last, s_last,
                                     v_last)
        comm_round += comm_round_func

        del client_manage.net_glob, client_manage.dataset
        del client_manage

        # do local training
        client_idx = np.random.choice(range(args.num_users), m, replace=False)
        client_manage = ClientManage(args, net_glob, client_idx, dataset_train,
                                     dict_users, hyper_param)
        x_glob, w_glob, v_glob, comm_round_func = client_manage.fed_locals(
            h_glob, q_glob, v_glob)
        comm_round += comm_round_func
        net_glob.load_state_dict(w_glob)
        for k in hyper_param:
            hyper_param[k].data = x_glob[k].data
        del w_glob, x_glob
        # print("Hyper param: ", hyper_param)

        # print round information
        print("Round {:3d}".format(iter))

        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train_real, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print(
            "Test acc/loss: {:.2f} {:.2f}".format(acc_test, loss_test),
            "Train acc/loss: {:.2f} {:.2f}".format(acc_train, loss_train),
            f"Comm round: {comm_round}",
        )

        logs.logging(client_idx, acc_test, acc_train, loss_test, loss_train,
                     comm_round)
        logs.save()

        ##### Garbage collection #####
        del client_manage.net_glob, client_manage.dataset
        del client_manage
        if p_last and s_last:
            del p_last[:], s_last[:]
            print("p_last and s_last deleted")
        del h_last, q_last, p_last, s_last, v_last
        h_last, q_last, p_last, s_last, v_last = (
            h_glob,
            q_glob,
            p_locals,
            s_locals,
            v_glob,
        )
        del h_glob, q_glob, p_locals, s_locals, v_glob
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data')
                                            and torch.is_tensor(obj.data)):
                    #print(type(obj), obj.size())
                    del obj
            except:
                pass
        torch.cuda.empty_cache()
        ##### Garbage collection #####

        r_final = torch.cuda.memory_allocated()
        print("Total Epoch GPU memory usage:", (r_final - r_init) / 1024**2,
              "MB")

        if args.round > 0 and comm_round > args.round:
            break
