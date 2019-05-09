import numpy as np
import cvxpy as cp
from cvxopt import matrix
from cvxopt.blas import dot 
from cvxopt.solvers import qp, options
from math import ceil
import sys


# ------- Formulation 1 ------
# min x^TΣx
# s.t. μ^Tx ≥ R
# 	   1^Tx = 1
#      x >= 0

def formulation1(t, n, mus, returns):
	
	mean_returns = np.mean(returns.T, axis=1)
	cov_matrix = np.cov(returns.T)

	Sigma = t * cov_matrix
	mu = t * np.reshape(mean_returns, (n,1))
	ones = np.ones((n,1))

	total_time = 0.0
	total_num_iters = 0

	xs = []
	for R in mus:
		x = cp.Variable(n)
		objective = cp.Minimize(cp.quad_form(x,Sigma))
		constraints = [mu.T@x >= R, ones.T@x == 1, x >= 0]
		prob = cp.Problem(objective, constraints)
		prob.solve()

		# print(prob.solver_stats.solve_time, prob.solver_stats.num_iters, R)
		
		total_time += prob.solver_stats.solve_time
		total_num_iters += prob.solver_stats.num_iters

		val = matrix(x.value)
		xs.append(val)

	avg_time = total_time / len(mus)
	avg_num_iters = ceil(total_num_iters / len(mus))
	
	return xs, avg_time, avg_num_iters



# ------- Formulation 2 ------
# max  μ^Tx 
# s.t  x^TΣx <= σ^2
#      1^Tx = 1
#      x >= 0

def formulation2(t, n, sigma_sq_list, returns):

	mean_returns = np.mean(returns.T, axis=1)
	cov_matrix = np.cov(returns.T)

	Sigma = t * cov_matrix
	mu = t * np.reshape(mean_returns, (n,1))
	ones = np.ones((n,1))

	total_time = 0.0
	total_num_iters = 0

	xs = []
	for sigma_sq in sigma_sq_list:
		x = cp.Variable(n)
		objective = cp.Maximize(mu.T@x)
		constraints = [cp.quad_form(x, Sigma) <= sigma_sq, ones.T@x == 1, x >= 0]
		prob = cp.Problem(objective, constraints)
		prob.solve()

		# print(prob.solver_stats.solve_time, prob.solver_stats.num_iters, sigma_sq)

		total_time += prob.solver_stats.solve_time
		total_num_iters += prob.solver_stats.num_iters

		val = matrix(x.value)
		xs.append(val)

	avg_time = total_time / len(sigma_sq_list)
	avg_num_iters = ceil(total_num_iters / len(sigma_sq_list))

	return xs, avg_time, avg_num_iters



# ------- Formulation 3 ------
# min  -μ^Tx + Rx^TΣx
# s.t  1^Tx = 1
#      x >= 0

def formulation3(t, n, mus, returns):
	
	mean_returns = np.mean(returns.T, axis=1)
	cov_matrix = np.cov(returns.T)

	Sigma = t * cov_matrix
	mu = t * np.reshape(mean_returns, (n,1))
	ones = np.ones((n,1))

	total_time = 0.0
	total_num_iters = 0

	xs = []
	for R in mus:
		x = cp.Variable(n)
		objective = cp.Minimize(R*cp.quad_form(x,Sigma) - mu.T@x)
		constraints = [ones.T@x == 1, x >= 0]
		prob = cp.Problem(objective, constraints)
		prob.solve()

		# print(prob.solver_stats.solve_time, prob.solver_stats.num_iters, R)
		
		total_time += prob.solver_stats.solve_time
		total_num_iters += prob.solver_stats.num_iters

		val = matrix(x.value)
		xs.append(val)

	avg_time = total_time / len(mus)
	avg_num_iters = ceil(total_num_iters / len(mus))
	
	return xs, avg_time, avg_num_iters



# ------- Formulation 4 ------
# min  ||Rx - rho.1||^2 
# s.t  mu^Tx = rho
#      1^Tx = 1
#      x >= 0

def formulation4(t, n, rho_list, returns):

	mean_returns = np.mean(returns.T, axis=1)
	cov_matrix = np.cov(returns.T)
	T, _ = returns.shape

	R = t * returns
	mu = t * np.reshape(mean_returns, (n,1))
	ones = np.ones((n,1))
	ones_ = np.ones((T,))

	total_time = 0.0
	total_num_iters = 0

	xs = []
	for rho in rho_list:
		x = cp.Variable(n)
		objective = cp.Minimize(cp.sum_squares(R*x - rho * ones_))
		constraints = [mu.T@x == rho, ones.T@x == 1, x >= 0]
		prob = cp.Problem(objective, constraints)
		prob.solve()

		# print(prob.solver_stats.solve_time, prob.solver_stats.num_iters, rho)

		total_time += prob.solver_stats.solve_time
		total_num_iters += prob.solver_stats.num_iters

		val = matrix(x.value)
		xs.append(val)

	avg_time = total_time / len(rho_list)
	avg_num_iters = ceil(total_num_iters / len(rho_list))

	return xs, avg_time, avg_num_iters