python main_imbalance_blo.py  --epoch 300  --round 100000 --lr 0.01 --hlr 0.02 \
--neumann 5 --inner_ep 10 \
--hvp_method global_batch --optim sgd \
--output output/im_noniid_fednest.yaml --q_noniid 0.1  --alpha 0.05 --beta 0.1 --frac 0.1 --momentum_rho 0.5

python main_imbalance_blo.py  --epoch 3000  --round 100000 --lr 0.01 --hlr 0.02 --neumann 5 --inner_ep 10 --local_ep 5 --outer_tau 1 --hvp_method global_batch --optim sgd --output output/im_noniid_fednest.yaml --q_noniid 0.2  --alpha 0.05 --beta 0.1 --frac 0.1 --momentum_rho 0.5

python main_imbalance.py  --epoch 300  --round 100000 --lr 0.01 --hlr 0.02 \
--neumann 3 --inner_ep 10 --local_ep 1 --outer_tau 10 \
--hvp_method global_batch --optim svrg --q_noniid 0.1 \
--output output/im_noniid_fednest.yaml 