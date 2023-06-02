import copy
from cv2 import log
import numpy as np

import torch

from utils.Fed import FedAvg, FedAvgGradient, FedAvgP
from core.SGDClient import SGDClient
from core.SVRGClient import SVRGClient
from core.BLOClient import BLOClient
from core.Client import Client
from core.function import assign_hyper_gradient, assign_weight_value


class ClientManage:

    def __init__(self, args, net_glob, client_idx, dataset, dict_users,
                 hyper_param) -> None:
        self.net_glob = net_glob
        self.client_idx = client_idx
        self.args = args
        self.dataset = dataset
        self.dict_users = dict_users

        self.hyper_param = copy.deepcopy(hyper_param)

    def fed_in(self):
        print(self.client_idx)
        w_glob = self.net_glob.state_dict()
        if self.args.all_clients:
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(self.args.num_users)]
        else:
            w_locals = []

        loss_locals = []
        grad_locals = []
        client_locals = []

        for idx in self.client_idx:
            if self.args.optim == "sgd":
                client = SGDClient(
                    self.args,
                    idx,
                    copy.deepcopy(self.net_glob),
                    self.dataset,
                    self.dict_users,
                    self.hyper_param,
                )
            elif self.args.optim == "svrg":
                client = SVRGClient(
                    self.args,
                    idx,
                    copy.deepcopy(self.net_glob),
                    self.dataset,
                    self.dict_users,
                    self.hyper_param,
                )
                grad = client.batch_grad()
                grad_locals.append(grad)
            else:
                raise NotImplementedError
            client_locals.append(client)
        if self.args.optim == "svrg":
            avg_grad = FedAvgGradient(grad_locals)
            for client in client_locals:
                client.set_avg_q(avg_grad)
        for client in client_locals:
            w, loss = client.train_epoch()
            if self.args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        self.net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        return w_glob, loss_avg

    def fedIHGP(self, client_locals):
        d_out_d_y_locals = []
        for client in client_locals:
            d_out_d_y = client.grad_d_out_d_y(create_graph=False)
            d_out_d_y.detach()
            d_out_d_y_locals.append(d_out_d_y)
        p = FedAvgP(d_out_d_y_locals, self.args)

        p_locals = []
        if self.args.hvp_method == "global_batch":
            for i in range(self.args.neumann):
                for client in client_locals:
                    p_client = client.hvp_iter(p, self.args.hlr)
                    p_locals.append(p_client)
                p = FedAvgP(p_locals, self.args)
        elif self.args.hvp_method == "local_batch":
            for client in client_locals:
                p_client = p.clone()
                for _ in range(self.args.neumann):
                    p_client = client.hvp_iter(p_client, self.args.hlr)
                p_locals.append(p_client)
            p = FedAvgP(p_locals, self.args)
        elif self.args.hvp_method == "seperate":
            for client in client_locals:
                d_out_d_y = client.grad_d_out_d_y()
                p_client = d_out_d_y.clone()
                for _ in range(self.args.neumann):
                    p_client = client.hvp_iter(p_client, self.args.hlr)
                p_locals.append(p_client)
            p = FedAvgP(p_locals, self.args)

        else:
            raise NotImplementedError
        return p

    def lfed_out(self, client_locals):
        hg_locals = []
        for client in client_locals:
            for _ in range(self.args.outer_tau):
                client.hyper_iter = 0
                d_out_d_y = client.grad_d_out_d_y()
                p_client = d_out_d_y.clone()
                for _ in range(self.args.neumann):
                    p_client = client.hvp_iter(p_client, self.args.hlr)
                hg_client = client.hyper_grad(p_client.clone())
                hg = client.hyper_update(hg_client)
            hg_locals.append(hg)
        hg_glob = FedAvgP(hg_locals, self.args)
        return hg_glob, 1

    def fed_out(self):
        client_locals = []
        for idx in self.client_idx:
            client = Client(
                self.args,
                idx,
                copy.deepcopy(self.net_glob),
                self.dataset,
                self.dict_users,
                self.hyper_param,
            )
            client_locals.append(client)

        if self.args.hvp_method == "seperate":
            return self.lfed_out(client_locals)

        # for client in client_locals:
        p = self.fedIHGP(client_locals)
        comm_round = 1 + self.args.neumann

        hg_locals = []
        for client in client_locals:
            hg = client.hyper_grad(p.clone())
            hg_locals.append(hg)
        hg_glob = FedAvgP(hg_locals, self.args)
        comm_round += 1
        hg_locals = []
        for client in client_locals:
            for _ in range(self.args.outer_tau):
                h = client.hyper_svrg_update(hg_glob)
            hg_locals.append(h)

        hg_glob = FedAvgP(hg_locals, self.args)
        comm_round += 1

        return hg_glob, comm_round

    def fed_globhq(self, rho, last_h, last_q, last_p, last_s, last_v):

        comm_round = 0
        client_locals = []
        for idx in self.client_idx:
            client = Client(
                self.args,
                idx,
                copy.deepcopy(self.net_glob),
                self.dataset,
                self.dict_users,
                self.hyper_param,
            )
            client_locals.append(client)

        if not last_s:
            print("Init global IHGP")
            v_glob = self.fedIHGP(client_locals)
            if self.args.hvp_method == "seperate":
                comm_round = 1
            elif self.args.hvp_method == "global_batch":
                comm_round = 1 + self.args.neumann
            elif self.args.hvp_method == "local_batch":
                comm_round = 2
            print("Done global IHGP")
        else:
            v_glob = last_v

        s_locals = []
        p_locals = []
        h_locals = []
        q_locals = []

        for i, client in enumerate(client_locals):
            p_locals.append(client.p_func(v_glob))
            h_locals.append(p_locals[-1])

            s_locals.append(client.s_func(v_glob))
            q_locals.append(s_locals[-1])
        print("Done local_sp_func")
        with torch.no_grad():
            if rho > 0 and (last_s is not None):
                print("Update local_hq with momentum")
                for i, client in enumerate(client_locals):
                    h_locals[i] = h_locals[i] + (1 - rho) * (last_h -
                                                             last_p[i])
                    q_locals[i][0] = q_locals[i][0] + (1 - rho) * (
                        last_q[0] - last_s[i][0])
                    q_locals[i][1] = q_locals[i][1] + (1 - rho) * (
                        last_q[1] - last_s[i][1])
            h_glob = FedAvgP(h_locals, self.args)
            q0 = [q[0] for q in q_locals]
            q1 = [q[1] for q in q_locals]
            q_glob = [
                FedAvgP(q0, self.args),
                FedAvgP(q1, self.args),
            ]
        comm_round += 1
        for client in client_locals:
            try:
                del client.net, client.ldr_train, client.ldr_val
                del client.hyper_param, client.hyper_param_init
                del client.args
                del client.net0
            except:
                pass
        del client_locals[:]
        del client_locals
        del h_locals[:], q_locals[:]
        del h_locals, q_locals
        del q0[:], q1[:]
        del q0, q1

        return h_glob, q_glob, p_locals, s_locals, v_glob, comm_round

    def fed_locals(self, h_glob, q_glob, v_glob):
        # record gpu usage

        client_locals = []
        print(self.client_idx)
        for idx in self.client_idx:
            client = BLOClient(
                self.args,
                idx,
                copy.deepcopy(self.net_glob),
                self.dataset,
                self.dict_users,
                self.hyper_param,
            )
            client_locals.append(client)

        hp_locals = []
        w_locals = []
        v_locals = []
        for client in client_locals:
            hp, (w, v) = client.update_local(h_glob, q_glob, v_glob)
            hp_locals.append(hp)
            w_locals.append(w)
            v_locals.append(v)
        hp_glob = FedAvgP(hp_locals, self.args)
        w_glob = FedAvgP(w_locals, self.args)
        v_glob = FedAvgP(v_locals, self.args)

        w_glob_dict_copy = copy.deepcopy(self.net_glob.state_dict())
        w_glob_dict_copy = assign_weight_value(w_glob_dict_copy, w_glob, False)

        hp_glob_dict = copy.deepcopy(self.hyper_param)
        hp_glob_dict = assign_weight_value(hp_glob_dict, hp_glob, True)

        del hp_locals, w_locals, v_locals, hp_glob, w_glob
        for client in client_locals:
            try:
                del client.net, client.ldr_train, client.ldr_val
                del client.hyper_param, client.hyper_param_init
                del client.net0
            except:
                pass
        del client_locals[:]
        del client_locals

        return hp_glob_dict, w_glob_dict_copy, v_glob, 1
        # return hp_glob, w_glob, v_glob, hp_glob_dict, w_glob_dict_copy
