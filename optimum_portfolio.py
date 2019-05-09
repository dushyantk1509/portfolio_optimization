# -*- coding: utf-8 -*- 

# All the codes are written by us except 
# data loading part refered from from http://stanford.edu/class/ee103/data/load_data.py
# plotting efficient frontier and generating random portfolios are refered from 
# https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
# which we have mentioned in repective functions in utils.py


import numpy as np
import pandas as pd
from math import sqrt
import sys
from utils import plot_uniform_portfolio_alloc, plot_returns, plot_risk_returns, plot_efficient_frontier, plot_random_simulated_portfolios, plot_best_portfolio_alloc, plot_efficient_frontier_together, get_avg_time
from formulations import formulation1, formulation2, formulation3, formulation4

# ------- loading data ------
# data loading is refered from http://stanford.edu/class/ee103/data/load_data.py

returns=pd.read_csv('returns.txt',index_col=0)
assets=np.array(returns.columns)
dates=np.array(returns.index)
returns=returns.as_matrix()

np.random.seed(500)

N = len(assets)
T = len(returns)

print("\nSome stats about the dataset.....")
print("No. of assets : ", N)
print("No. of days : ", T)
print("Size of returns : ", returns.shape)

# assets
# ['AGG' 'DBC' 'DFE' 'DIA' 'DXJ' 'EEM' 'EFA' 'EWG' 'EWH' 'EWI' 'EWT' 'EWU'
#  'EWW' 'EWY' 'EWZ' 'EZU' 'FEZ' 'FXI' 'GDX' 'GLD' 'IAU' 'IBB' 'ITB' 'IVV'
#  'IWD' 'IWM' 'IYR' 'KBE' 'KRE' 'LQD' 'OIL' 'SDS' 'SH' 'SLV' 'SPY' 'USO'
#  'VGK' 'VNQ' 'VTI' 'VWO' 'XHB' 'XLB' 'XLE' 'XLF' 'XLI' 'XLK' 'XLP' 'XLU'
#  'XLV' 'XLY' 'XME' 'XOP' 'USDOLLAR']

# 'USDOLLAR' is risk-free asset

# ------- Initializing some fixed consts ------
n = 5   							# for the purpose of experimentation we are considering only 5 random assets
t = 250 							# no. of trading days per year
V = 10000							# Initially assumed portfolio investment value
num_portfolios = 50000				# how many portfolios to generate randomly
risk_free_rate = 0.00084142			# required for sharpe ratio calculation calculated from risk free assets


# ------- Calculating mean returns and risk of assets ------
mean_returns = np.mean(returns, axis=0)
mean_returns = mean_returns.reshape((mean_returns.shape[0], 1))
print("No. of assets with positive mean returns : ", np.sum(mean_returns>0))

assets_idx = [0, 7, 33, 44, 52]					# assets index considere for experimentation

print("Assets considered are : ", assets_idx, assets[assets_idx])
print("Annual returns of assets : ", t*np.squeeze(mean_returns[assets_idx]))

mean_returns_exp = mean_returns[assets_idx]		# mean returns of assets considered for experimentation
returns_exp = returns[:,assets_idx]

cov_matrix_exp = np.cov(returns_exp.T)
print("Annual volatility of assets : ", np.sqrt(t)*np.array([sqrt(cov_matrix_exp[i,i]) for i in range(0,n)]))

# ------- Some plots ------
print("\nProducing some plots.....\n")
plot_uniform_portfolio_alloc(V, n, returns_exp, assets, assets_idx)						# plotting uniform allocation portfolio values
plot_returns(n, returns, assets, assets_idx)											# plotting returns of various assets on a daily basis
plot_risk_returns(n, t, returns, returns_exp)											# plotting annualized risk-return curve
plot_random_simulated_portfolios(n, t, num_portfolios, risk_free_rate, returns_exp)		# plotting random simulated portfolios risk and returns

# ------- Obtaining efficient frontiers ------

# # ------- Formulation 1 ------
# # min x^TΣx
# # s.t. μ^Tx ≥ R
# # 	   1^Tx = 1
# #      x >= 0
print("\nObtaining efficient frontier and stats for formulation 1.........")
mus = [0.001 * mu for mu in range(0, 85, 13)] + [0.082, 0.08256]						# carefully choosing the values
xs1, avg_time1, avg_num_iters1 = formulation1(t, n, mus, returns_exp)
print("Average time to solve formulation 1 : " + str(get_avg_time(t, n, mus, returns_exp, 1) * 1000000) + " micro seconds")
print("Average number of iterations to solve formulation 1 : ", avg_num_iters1)
plot_efficient_frontier(xs1, n, t, num_portfolios, risk_free_rate, returns_exp, 1, color_code='m')		# plotting efficient frontier for this formulation


# # ------- Formulation 2 ------
# # max  μ^Tx 
# # s.t  x^TΣx <= σ^2
# #      1^Tx = 1
# #      x >= 0
print("\nObtaining efficient frontier and stats for formulation 2.........")
sigma_sq_list = [0.00084, 0.019, 0.054, 0.096, 0.150, 0.196, 0.223, 0.284, 0.316]		# carefully choosing the values
print(sigma_sq_list)
sigma_sq_list = [ sigma_sq**2 for sigma_sq in sigma_sq_list]
xs2, avg_time2, avg_num_iters2 = formulation2(t, n, sigma_sq_list, returns_exp)
print("Average time to solve formulation 2 : " + str(get_avg_time(t, n, sigma_sq_list, returns_exp, 2) * 1000000) + " micro seconds")
print("Average number of iterations to solve formulation 2 : ", avg_num_iters2)
plot_efficient_frontier(xs2, n, t, num_portfolios, risk_free_rate, returns_exp, 2, color_code='g')		# plotting efficient frontier for this formulation


# # ------- Formulation 2 ------
# # min  -μ^Tx + Rx^TΣx
# # s.t  1^Tx = 1
# #      x >= 0
print("\nObtaining efficient frontier and stats for formulation 3.........")
mus = [ 10**(5.0*t/30-1.0) for t in range(1, 30) ]										# randomly choosing the values
xs3, avg_time3, avg_num_iters3 = formulation3(t, n, mus, returns_exp)
print("Average time to solve formulation 3 : " + str(get_avg_time(t, n, mus, returns_exp, 3) * 1000000) + " micro seconds")
print("Average number of iterations to solve formulation 3 : ", avg_num_iters3)
plot_efficient_frontier(xs3, n, t, num_portfolios, risk_free_rate, returns_exp, 3, color_code='r')		# plotting efficient frontier for this formulation


# # ------- Formulation 4 ------
# # min  ||Rx - rho.1||^2 
# # s.t  mu^Tx = rho
# #      1^Tx = 1
# #      x >= 0
print("\nObtaining efficient frontier and stats for formulation 4.........")
mus = [0.001 * mu for mu in range(10, 90, 10)] + [0.081, 0.0815]						# carefully choosing the values
xs4, avg_time4, avg_num_iters4 = formulation4(t, n, mus, returns_exp)
print("Average time to solve formulation 4 : " + str(get_avg_time(t, n, mus, returns_exp, 4) * 1000000) + " micro seconds")
print("Average number of iterations to solve formulation 4 : ", avg_num_iters4)
plot_efficient_frontier(xs4, n, t, num_portfolios, risk_free_rate, returns_exp, 4, color_code='b')		# plotting efficient frontier for this formulation


# # ------- Some more plotting ------
print("\nProducing some more plots.....")
mus=[0.05, 0.08]
xs, _, _ = formulation1(t, n, mus, returns_exp)
plot_best_portfolio_alloc(xs, mus, V, n, returns_exp, assets, assets_idx)		# plotting optimum portfolio values for 2 different mus

color_code = ['m', 'g', 'r', 'b']
plot_efficient_frontier_together(xs1, xs2, xs3, xs4, n, t, num_portfolios, risk_free_rate, returns_exp, color_code)		# plotting efficient frontier for this all formulations together