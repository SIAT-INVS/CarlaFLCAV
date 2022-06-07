from calendar import c
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import random

# task-1 CNN, task-2 YOLO, task-3 SECOND
a = [11.84, 8.15, 0.51]
b = [1.23, 0.62, 0.10]
a = [a[i]+0.05*random.uniform(0,1)*a[i] for i in range(len(a))] # add noise due to extrapolation
b = [b[i]+0.05*random.uniform(0,1)*b[i] for i in range(len(b))]
D = [0.7, 0.7, 1.6] # MB
DNNSIZE = [23, 14, 60] # MB

# problem data.
alpha = [0.2, 0.2, 0.2] # domain shift
beta = [0.1, 0.05, 0.5] # fed degragation
gamma = [0.1, 0.05, 0.5] # fed degragation
n = 3 # number of tasks
s = 3 # number of stages
y = [4*500, 4*1000, 4*100] # number of samples at edge
z = [3*500, 3*1000, 3*100] # number of samples at remote edge
wireless_budget = 4096 # 4MB
wireline_budget = 4096 # 4MB

# variables
x = cp.Variable(n)
r = cp.Variable(n)
t = cp.Variable(n)
u = cp.Variable((s,n))
v = cp.Variable((s,n))
slack = cp.Variable(n)
# constraints
basic_contraints = [0 <= x, 0 <=r, 0 <= t, 0 <= u, 0 <= v]
budget_constraints = [cp.sum(cp.sum(u)) <= wireless_budget, cp.sum(cp.sum(v)) <= wireline_budget]
x_constraints = [x[i] <= u[0][i]/D[i] for i in range(n)]
x_constraints += [x[i] <= v[0][i]/D[i] for i in range(n)]
r_constraints = [r[i] <= u[1][i]/(3*DNNSIZE[i]) for i in range(n)]
t_constraints = [t[i] <= v[2][i]/(3*DNNSIZE[i]) for i in range(n)]
slack_constraints = [slack[i] >= a[i] * cp.power(alpha[i] * x[i] \
                    + beta[i] * y[i] * (1 - cp.inv_pos(r[i]+1)) \
                    + gamma[i] * z[i] * (1 - cp.inv_pos(t[i]+1)), -b[i]) for i in range(n)]

constraints = []
constraints += basic_contraints
constraints += budget_constraints
constraints += x_constraints
constraints += r_constraints
constraints += t_constraints
constraints += slack_constraints

objective = cp.Minimize(cp.sum(slack))
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