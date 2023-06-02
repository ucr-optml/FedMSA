import torch
import copy
from core.Client import Client
from core.function import (
    assign_hyper_gradient,
    gather_flat_hyper_params,
    assign_weight_value,
    gather_flat_weight,
)


class BLOClient(Client):

    def __init__(self,
                 args,
                 client_id,
                 net,
                 dataset=None,
                 idxs=None,
                 hyper_param=None) -> None:
        super().__init__(args, client_id, net, dataset, idxs, hyper_param)

    def update_local(self, h_glob, q_glob, v_glob):
        self.net.train()
        # train and update
        h_last, q_last = h_glob, q_glob

        x_last = gather_flat_hyper_params(self.hyper_param)
        x_cur = x_last

        v_last = v_glob.clone()
        v_cur = v_last

        w_last = gather_flat_weight(self.net.state_dict())
        w_cur = w_last.clone()

        p_last = self.p_func(v_last)
        s_last = self.s_func(v_last)
        for _ in range(self.args.inner_ep):
            p_cur = self.p_func(v_cur)
            s_cur = self.s_func(v_cur)
            h_cur = p_cur + h_last - p_last
            q_cur = [
                s_cur[0] + q_last[0] - s_last[0],
                s_cur[1] + q_last[1] - s_last[1]
            ]

            # update x, w, v
            x_next = x_cur - self.args.alpha * h_cur
            v_next = v_cur - self.args.beta * q_cur[1]
            w_next = w_cur - self.args.beta * q_cur[0]
            # assign w, x back to net and hyper_param
            net_state_dict = self.net.state_dict()
            net_state_dict = assign_weight_value(net_state_dict, w_next, False)
            self.net.load_state_dict(net_state_dict)

            hyper_para = assign_weight_value(self.hyper_param, x_next, True)
            for key in hyper_para:
                self.hyper_param[key].data = hyper_para[key].data

            del v_last, w_last, x_last, p_last, s_last, h_last, q_last
            v_last, v_cur = v_cur, v_next
            w_last, w_cur = w_cur, w_next
            x_last, x_cur = x_cur, x_next
            p_last = p_cur
            s_last = s_cur
            h_last = h_cur
            q_last = q_cur
        return x_cur, (w_cur, v_cur)
