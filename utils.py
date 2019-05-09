import numpy as np
import pandas as pd
from random import *
import random
from cvxopt import matrix
from cvxopt.blas import dot
from math import sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm
from formulations import formulation1, formulation2, formulation3, formulation4

np.random.seed(500)

def plot_uniform_portfolio_alloc(V, n, returns, assets, assets_idx):

	T = len(returns)
	w = 1.0/n * np.ones((n,1))
	r = np.matmul(returns, w)

	commulative_returns = []
	s = 0

	for i in range(T):
		V_t = V + V*s
		s += r[i]
		commulative_returns.append(V_t)

	x = [days for days in range(0,T)]

	asset1 = []
	asset2 = []
	asset3 = []
	asset4 = []
	asset5 = []

	s1 = 0
	s2 = 0
	s3 = 0
	s4 = 0
	s5 = 0

	for i in range(T):
		V_t_1 = V + V*s1
		V_t_2 = V + V*s2
		V_t_3 = V + V*s3
		V_t_4 = V + V*s4
		V_t_5 = V + V*s5

		s1 += returns[i, 0]
		s2 += returns[i, 1]
		s3 += returns[i, 2]
		s4 += returns[i, 3]
		s5 += returns[i, 4]

		asset1.append(V_t_1)
		asset2.append(V_t_2)
		asset3.append(V_t_3)
		asset4.append(V_t_4)
		asset5.append(V_t_5)

	plt.figure()
	plt.plot(x, asset1, linewidth=1.0, label = assets[assets_idx[0]])
	plt.plot(x, asset2, linewidth=1.0, label = assets[assets_idx[1]])
	plt.plot(x, asset3, linewidth=1.0, label = assets[assets_idx[2]])
	plt.plot(x, asset4, linewidth=1.0, label = assets[assets_idx[3]])
	plt.plot(x, asset5, linewidth=1.0, label = assets[assets_idx[4]])
	plt.plot(x, commulative_returns, linewidth=1.0, label = "uniform allocation")
	plt.legend()
	plt.xlabel('Days')
	plt.ylabel('Amount')
	plt.savefig("uniform_alloc.png")



def plot_returns(n, returns, assets, assets_idx):

	T = len(returns)
	x = [days for days in range(0,T)]
	plt.figure()
	for i in range(0,n):
		plt.plot(x, returns[:, assets_idx[i]], linewidth=1.0, label = assets[assets_idx[i]])

	plt.xlabel('days')
	plt.ylabel('returns')
	plt.legend()
	plt.savefig("daily_returns.png")



def plot_risk_returns(n, t, returns, returns_exp):

	mean_returns = np.mean(returns.T, axis=1)
	cov_matrix = np.cov(returns.T)
	T, N = returns.shape

	plt.figure()
	plt.scatter(sqrt(t)*np.array([sqrt(cov_matrix[i,i]) for i in range(0,N)]), t*mean_returns, s=3, c='g')

	mean_returns = np.mean(returns_exp.T, axis=1)
	cov_matrix = np.cov(returns_exp.T)
	x = sqrt(t)*np.array([sqrt(cov_matrix[i,i]) for i in range(0,n)])
	y = t*mean_returns
	point_label = ['AGG', 'EWG', 'SLV', 'XLI']

	plt.scatter(x, y, s=20, c='m')
	for i, txt in enumerate(point_label):
		plt.annotate(txt, (x[i], y[i]))

	plt.xlim([0.0, 0.5])
	plt.xlabel('Annualized risk')
	plt.ylabel('Annualized returns')
	plt.savefig("risk_return.png")



def generate_random_portfolios(n, t, num_portfolios, risk_free_rate, returns):
	# This function if refered from 
	# https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f

	mean_returns = np.mean(returns.T, axis=1)
	cov_matrix = np.cov(returns.T)
	results = np.zeros((3,num_portfolios))

	for i in range(num_portfolios):
		weights = np.random.random(n)
		weights /= np.sum(weights)
		returns = np.sum(mean_returns*weights ) * t
		std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(t)

		results[0,i] = std
		results[1,i] = returns
		results[2,i] = (returns - risk_free_rate) / std

	return results



def plot_random_simulated_portfolios(n, t, num_portfolios, risk_free_rate, returns):

	results = generate_random_portfolios(n, t, num_portfolios, risk_free_rate, returns)

	plt.figure()
	plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=20, alpha=0.3, label='Random portfolios')
	plt.xlabel('annualised risk')
	plt.ylabel('annualised returns')
	plt.legend()
	plt.savefig("random_porfolio.png")



def plot_best_portfolio_alloc(xs, mus, V, n, returns, assets, assets_idx):

	T = len(returns)
	w = 1.0/n * np.ones((n,1))
	r = np.matmul(returns, w)

	x = [days for days in range(0,T)]

	asset1 = []
	asset2 = []
	asset3 = []
	asset4 = []
	asset5 = []

	s1 = 0
	s2 = 0
	s3 = 0
	s4 = 0
	s5 = 0

	for i in range(T):
		V_t_1 = V + V*s1
		V_t_2 = V + V*s2
		V_t_3 = V + V*s3
		V_t_4 = V + V*s4
		V_t_5 = V + V*s5

		s1 += returns[i, 0]
		s2 += returns[i, 1]
		s3 += returns[i, 2]
		s4 += returns[i, 3]
		s5 += returns[i, 4]

		asset1.append(V_t_1)
		asset2.append(V_t_2)
		asset3.append(V_t_3)
		asset4.append(V_t_4)
		asset5.append(V_t_5)

	plt.figure()
	plt.plot(x, asset1, linewidth=1.0, label = assets[assets_idx[0]])
	plt.plot(x, asset2, linewidth=1.0, label = assets[assets_idx[1]])
	plt.plot(x, asset3, linewidth=1.0, label = assets[assets_idx[2]])
	plt.plot(x, asset4, linewidth=1.0, label = assets[assets_idx[3]])
	plt.plot(x, asset5, linewidth=1.0, label = assets[assets_idx[4]])

	for i in range(len(mus)):
		w = xs[i]
		w = np.reshape(w, (n,1))

		r = np.matmul(returns, w)
		commulative_returns = []
		s = 0

		for j in range(T):
			V_t = V + V*s
			s += r[j]
			commulative_returns.append(V_t)

		plt.plot(x, commulative_returns, linewidth=1.0, label = "mean return - " + str(mus[i]))

	plt.legend()
	plt.xlabel('Days')
	plt.ylabel('Amount')
	plt.savefig("optimum_alloc.png")



def plot_efficient_frontier(xs, n, t, num_portfolios, risk_free_rate, returns, formulation_no, color_code):
	# This function mainly line# 228 is refered from 
	# https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f 

	mean_returns = np.mean(returns.T, axis=1)
	cov_matrix = np.cov(returns.T)
	results = generate_random_portfolios(n, t, num_portfolios, risk_free_rate, returns)

	plt.figure()
	plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=20, alpha=0.3, label='Random portfolios')

	p = matrix(np.reshape(mean_returns, (1,n)))
	Q = matrix(cov_matrix, (n,n),  'd')
	returns = [ t * dot(p,x) for x in xs ]
	risks = [ sqrt(t) * sqrt(dot(x, Q*x)) for x in xs ]

	# print(risks)
	# print(returns)

	plt.plot(risks, returns, linewidth=0.5, marker='o', c=color_code, label = 'Efficient Frontier')
	plt.legend()
	plt.xlabel('Annualized risk')
	plt.ylabel('Annualized returns')
	plt.savefig("form" + str(formulation_no) + ".png")



def plot_efficient_frontier_together(xs1, xs2, xs3, xs4, n, t, num_portfolios, risk_free_rate, returns, color_code):
	# This function mainly line# 255 is refered from 
	# https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f

	mean_returns = np.mean(returns.T, axis=1)
	cov_matrix = np.cov(returns.T)
	results = generate_random_portfolios(n, t, num_portfolios, risk_free_rate, returns)

	plt.figure()
	plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=20, alpha=0.3, label='Random portfolios')

	for i, xs in enumerate([xs1, xs2, xs3, xs4]):
		p = matrix(np.reshape(mean_returns, (1,n)))
		Q = matrix(cov_matrix, (n,n),  'd')
		returns = [ t * dot(p,x) for x in xs ]
		risks = [ sqrt(t) * sqrt(dot(x, Q*x)) for x in xs ]

		plt.plot(risks, returns, linewidth=0.5, marker='o', c=color_code[i], label = 'Efficient Frontier')
	
	plt.legend()
	plt.xlabel('Annualized risk')
	plt.ylabel('Annualized returns')
	plt.savefig("all.png")

def get_avg_time(t, n, mus, returns, formulation_no):

	total_time = 0.0
	N = 100

	for i in tqdm(range(N)):
		if formulation_no == 1:
			_, time, _ = formulation1(t, n, mus, returns)
		elif formulation_no == 2:
			_, time, _ = formulation2(t, n, mus, returns)
		elif formulation_no == 3:
			_, time, _ = formulation3(t, n, mus, returns)
		elif formulation_no == 4:
			_, time, _ = formulation4(t, n, mus, returns)

		total_time += time

	avg_time = total_time / N

	return avg_time