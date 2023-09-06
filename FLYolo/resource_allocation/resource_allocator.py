from calendar import c
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class Resource_Allocator:
    def __init__(self):
        # task-1 CNN, task-2 YOLO, task-3 SECOND
        self.a = [11.84, 8.15, 0.51]
        self.b = [1.23, 0.62, 0.10]
        self.D = [0.7, 0.7, 1.6] # MB
        self.DNNSIZE = [23, 14, 60] # MB
        self.alpha = [0.1, 0.1, 0.1] # domain shift
        self.beta = [0.5, 0.5, 0.5] # fed degragation
        self.n = 3 # number of tasks
        self.s = 3 # number of stages
        self.y = [4*500, 4*1000, 4*100] # number of samples at edge Town03
        self.z = [3*500, 3*1000, 3*100] # number of samples at edge Town05
        self.INTERVAL = 1
        self.task_weights = [1/3, 1/3, 1/3]
        # wireless_budget = 4096 # 4MB
        # wireline_budget = 4096 # 4MB

    def allocate(self, wireless_budget, wireline_budget):
        a = self.a 
        b = self.b
        D = self.D
        DNNSIZE = self.DNNSIZE
        alpha = self.alpha
        beta = self.beta
        n = self.n  # number of tasks
        s = self.s  # number of stages
        y = self.y
        z = self.z
        INTERVAL = self.INTERVAL
        num_veh = 7
        num_edge = 2
        
        # variables
        x = cp.Variable(n)
        r = cp.Variable(n)
        t = cp.Variable(n)
        u = cp.Variable((s,n))
        v = cp.Variable((s,n))
        slack = cp.Variable(n)
        # constraints
        basic_contraints = [100 <= x, 0 <=r, 0 <= t, 0 <= u, 0 <= v]
        budget_constraints = [cp.sum(cp.sum(u)) <= wireless_budget, cp.sum(cp.sum(v)) <= wireline_budget]
        x_constraints = [x[i] <= u[0][i]/D[i] for i in range(n)]
        x_constraints += [x[i] <= v[0][i]/D[i] for i in range(n)]
        r_constraints = [r[i] <= u[1][i]/(num_veh*DNNSIZE[i]) for i in range(n)]
        t_constraints = [t[i] <= v[2][i]/(num_edge*DNNSIZE[i]) for i in range(n)]
        rt_constraints = [r[i] == INTERVAL * t[i] for i in range(n)]
        slack_constraints = [slack[i] >= a[i] * cp.power(alpha[i] * x[i] \
                + beta[i] * (y[i] + z[i]) * (1 - cp.inv_pos(t[i]+1)), -b[i]) for i in range(n)]

        constraints = []
        constraints += basic_contraints
        constraints += budget_constraints
        constraints += x_constraints
        constraints += r_constraints
        constraints += t_constraints
        constraints += rt_constraints
        constraints += slack_constraints

        objective = cp.Minimize(cp.sum(self.task_weights @ slack))
        # The multi-modal multi-stage joint optimization
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        print('The number of samples are:')
        print(x.value)
        print('The number of edge FL rounds are:')
        print(r.value)
        print('The number of cloud FL rounds are:')
        print(t.value)
        print('Wireless resource allocation:')
        print(u.value)
        print('Wireline resource allocation:')
        print(v.value)

        cnn_opt_array = np.zeros((3, 1))
        cnn_opt_array[0] = math.floor(x.value[0])
        cnn_opt_array[1] = math.floor(r.value[0])
        cnn_opt_array[2] = math.floor(t.value[0])

        yolo_opt_array = np.zeros((3, 1))
        yolo_opt_array[0] = math.floor(x.value[1])
        yolo_opt_array[1] = math.floor(r.value[1])
        yolo_opt_array[2] = math.floor(t.value[1])

        second_opt_array = np.zeros((3, 1))
        second_opt_array[0] = math.floor(x.value[2])
        second_opt_array[1] = math.floor(r.value[2])
        second_opt_array[2] = math.floor(t.value[2])

        return [cnn_opt_array, yolo_opt_array, second_opt_array]

