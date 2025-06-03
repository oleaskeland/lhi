#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:31:04 2020

@author: zen
"""

import numpy as np
import matplotlib.pyplot as plt

import learner as ln
from learner.integrator.hamiltonian import SV

class TBData(ln.Data):
    '''Data for learning the three body system.
    '''
    def __init__(self, h, train_traj_num, test_traj_num, train_num, test_num, add_h=False):
        super(TBData, self).__init__()
        self.solver = SV(None, self.dH, iterations=1, order=6, N=100)
        self.h = h
        self.train_traj_num = train_traj_num
        self.test_traj_num = test_traj_num
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        self.__init_data()
        
    @property
    def dim(self):
        return 12
    
    def dH(self, p, q):
        p1 = p[..., :2]
        p2 = p[..., 2:4]
        p3 = p[..., 4:6]
        q1 = q[..., :2]
        q2 = q[..., 2:4]
        q3 = q[..., 4:6]
        dHdp1 = p1
        dHdp2 = p2
        dHdp3 = p3
        dHdq1 = (q1-q2)/np.sum((q1-q2)**2, axis = -1, keepdims = True)**1.5 + (q1-q3)/np.sum((q1-q3)**2, axis = -1, keepdims = True)**1.5
        dHdq2 = (q2-q3)/np.sum((q2-q3)**2, axis = -1, keepdims = True)**1.5 + (q2-q1)/np.sum((q2-q1)**2, axis = -1, keepdims = True)**1.5
        dHdq3 = (q3-q1)/np.sum((q3-q1)**2, axis = -1, keepdims = True)**1.5 + (q3-q2)/np.sum((q3-q2)**2, axis = -1, keepdims = True)**1.5
        dHdp = np.hstack([dHdp1, dHdp2, dHdp3])
        dHdq = np.hstack([dHdq1, dHdq2, dHdq3])
        return dHdp, dHdq   
    
    def __generate_flow(self, x0, h, num):
        X = self.solver.flow(np.array(x0), h, num)
        x, y = X[:,:-1], X[:,1:]
        if self.add_h:
            x = np.concatenate([x, self.h * np.ones([x.shape[0], x.shape[1], 1])], axis = 2)
        return x, y
    
    def rotate2d(self, p, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s],[s, c]])
        R = np.transpose(R)
        return p.dot(R)
    
    def random_config(self, n, nu=2e-1, min_radius=0.9, max_radius=1.2, return_tensors=True):
        
        q1 = np.zeros([n, 2])
    
        q1 = 2*np.random.rand(n, 2) - 1
        r = np.random.rand(n) * (max_radius-min_radius) + min_radius
    
        ratio = r/np.sqrt(np.sum((q1**2), axis=1))
        q1 *= np.tile(np.expand_dims(ratio, 1), (1, 2))
        q2 = self.rotate2d(q1, theta=2*np.pi/3)
        q3 = self.rotate2d(q2, theta=2*np.pi/3)
    
        # # velocity that yields a circular orbit
        v1 = self.rotate2d(q1, theta=np.pi/2)
        v1 = v1 / np.tile(np.expand_dims(r**1.5, axis=1), (1, 2))
        v1 = v1 * np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2)) # scale factor to get circular trajectories
        v2 = self.rotate2d(v1, theta=2*np.pi/3)
        v3 = self.rotate2d(v2, theta=2*np.pi/3)
    
        # make the circular orbits slightly chaotic
        v1 *= 1 + nu*(2*np.random.rand(2) - 1)
        v2 *= 1 + nu*(2*np.random.rand(2) - 1)
        v3 *= 1 + nu*(2*np.random.rand(2) - 1)
    
        q = np.zeros([n, 6])
        p = np.zeros([n, 6])
    
        q[:, :2] = q1
        q[:, 2:4] = q2
        q[:, 4:] = q3
        p[:, :2] = v1
        p[:, 2:4] = v2
        p[:, 4:] = v3
  
        return np.hstack([p, q])
    
    def __init_data(self):
        np.random.seed(0)
        x0 = self.random_config(self.train_traj_num + self.test_traj_num)
        X_train, y_train = self.__generate_flow(x0[:self.train_traj_num], self.h, self.train_num)
        X_test, y_test = self.__generate_flow(x0[self.train_traj_num:], self.h, self.test_num)
        self.X_train = X_train.reshape([self.train_num*self.train_traj_num, -1])
        self.y_train = y_train.reshape([self.train_num*self.train_traj_num, -1])
        self.X_test = X_test.reshape([self.test_num*self.test_traj_num, -1])
        self.y_test = y_test.reshape([self.test_num*self.test_traj_num, -1])


def plot(data, net, net_type, fname="three_body_panel.png"):
    """
    Create a 1×3 panel: trajectories, energy H(t), and global MSE.
    Prints total trajectory MSE. Annotates param count on MSE plot.
    Extended from the original three_body.py to include energy and MSE plots.
    """
    # --- 1. generate true & predicted flows -------------------------------
    h_true = data.h / 10
    test_num_true = (data.test_num - 1) * 10

    if isinstance(net, ln.nn.HNN):
        flow_true = data.solver.flow(data.X_test_np[0][:-1], h_true, test_num_true)
        flow_pred = net.predict(data.X_test[0][:-1], data.h, data.test_num - 1,
                                keepinitx=True, returnnp=True)
    else:
        flow_true = data.solver.flow(data.X_test_np[0], h_true, test_num_true)
        flow_pred = net.predict(data.X_test[0], data.test_num - 1,
                                keepinitx=True, returnnp=True)

    flow_true_sub = flow_true[::10]

    # --- 2. compute energy H(t) -------------------------------------------
    def energy(flow):
        p, q = flow[:, :6], flow[:, 6:]
        p1, p2, p3 = p[:, :2], p[:, 2:4], p[:, 4:]
        q1, q2, q3 = q[:, :2], q[:, 2:4], q[:, 4:]
        T = 0.5 * (np.sum(p1**2, 1) + np.sum(p2**2, 1) + np.sum(p3**2, 1))
        V = (1 / np.sqrt(np.sum((q1 - q2)**2, 1)) +
             1 / np.sqrt(np.sum((q1 - q3)**2, 1)) +
             1 / np.sqrt(np.sum((q2 - q3)**2, 1)))
        return T - V

    E_true = energy(flow_true_sub)
    E_pred = energy(flow_pred)

    # --- 3. global per-step MSE -------------------------------------------
    mse = np.mean((flow_pred - flow_true_sub)**2, axis=1)
    total_mse = np.mean(mse)

    # --- 4. param count ---------------------------------------------------
    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # --- 5. Plot panel ----------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 18))

    # 1 Trajectories
    ax[0].plot(flow_true[:, 6], flow_true[:, 7], 'b', label='True')
    ax[0].plot(flow_true[:, 8], flow_true[:, 9], 'b')
    ax[0].plot(flow_true[:, 10], flow_true[:, 11], 'b')
    ax[0].scatter(flow_pred[:, 6], flow_pred[:, 7], c='r', s=10, label='Pred')
    ax[0].scatter(flow_pred[:, 8], flow_pred[:, 9], c='r', s=10)
    ax[0].scatter(flow_pred[:, 10], flow_pred[:, 11], c='r', s=10)
    ax[0].set_title(f"Trajectories ({net_type})")
    ax[0].legend(loc='lower left')

    # 2 Energy
    t = np.arange(len(E_true)) * data.h
    ax[1].plot(t, E_true, label='True')
    ax[1].plot(t, E_pred, '--', label='Pred')
    ax[1].set_title(f"Total Energy $H(t)$ ({net_type})")
    ax[1].set_xlabel("t")
    ax[1].legend()

    # 3 MSE
    ax[2].plot(t, mse)
    ax[2].set_title(f"Global Trajectory Error\n({net_type}, {param_count} params)")
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("MSE")

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

    print(f"Total MSE across trajectory for {net_type}: {total_mse:.4e}")



def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    h = 0.5
    train_num = 10
    test_num = 10
    train_traj_num = 400
    test_traj_num = 100
    # net
    net_type = 'LHI' # 'LA' or 'G' or 'HNN'
    LAlayers = 20
    LAsublayers = 4
    Glayers = 20
    Gwidth = 50
    activation = 'sigmoid'
    Hlayers = 6
    Hwidth = 50
    Hactivation = 'tanh'
    # training
    lr = 0.001
    iterations = 20000
    print_every = 1000

    
    add_h = True if net_type == 'HNN' else False
    criterion = None if net_type == 'HNN' else 'MSE'
    data = TBData(h, train_traj_num, test_traj_num, train_num, test_num, add_h)
    if net_type == 'LA':
        net = ln.nn.LASympNet(data.dim, LAlayers, LAsublayers, activation)
    elif net_type == 'G-SympNet':
        net = ln.nn.GSympNet(data.dim, Glayers, Gwidth, activation)
    elif net_type == 'LHI':
        net = ln.nn.LHI(dim=data.dim, shears=4, hidden_dim=20, h=h)
    elif net_type == 'HNN':
        net = ln.nn.HNN(data.dim, Hlayers, Hwidth, Hactivation)



    params = [p for p in net.parameters() if p.requires_grad]
    print(f"Total trainable parameters in {net_type}: {sum(p.numel() for p in params)}")
    args = {
        'data': data,
        'net': net,
        'criterion': criterion,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        "experiment_name": f'3B_{h}_{net_type}',
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    plot(data, ln.Brain.Best_model(), net_type=net_type, fname=f"three_body_panel_{net_type}.png")
    
if __name__ == '__main__':
    main()
