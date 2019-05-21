#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import linalg as nla
import scipy as sp
from scipy import linalg as sla
from time import time
from matplotlib import pyplot as plt
import os

def NewtonPoly(A, X0 = np.NAN, maxiter = 100, tol = np.NAN):
    if np.isnan(X0): # X0가 주어지지 않았을 때 m by m zero 행렬 처리
        X0 = np.zeros((A.shape[1],A.shape[2]))
    
    if A.shape[1] != A.shape[2]: # A가 square matrices의 모음이 아닐 때 예외처리
        raise ValueError('A가 정방행렬이 아닙니다.')
        
    m, n = A.shape[1], A.shape[0]-1 # m, n 초기화
    
    if np.isnan(tol): # tol이 주어지지 않았을 때 초기화
        tol = m * 1e-15
        
    Xs = [X0] # Xs는 X들을 담은 리스트로 초기화
    P_Xs, Hs, errs = [], [], [] # P_X, H, err 리스트 초기화
    S = np.zeros((A.shape[1],A.shape[2]))
    
    iter = 0
    err = 1e10 # error 초기화
    
    # Newton Iteration 시작
    while (err > tol) and (iter < maxiter):
        P_X = np.zeros((m**2, m**2))
        for k in range(1,n+1):
            for l in range(k):
                P_X = P_X + np.kron(nla.matrix_power(X0.transpose(),k-l-1), A[k,:,:] @ nla.matrix_power(X0, l))
        P_Xs.append(P_X) # P_X_i 저장
        vP = np.reshape(Pnomial(X0, A), m**2, 'F') # Reshape는 반드시 Fortran 방식으로
        h = nla.solve(P_X, vP)
        H = np.reshape(h, (m, m), order = 'F')
        Hs.append(H) # H_i 저장
        X0 = X0 - H # Newton Sequence 적용
        Xs.append(X0) # X_i 저장
        
        err = nla.norm(Pnomial(X0, A), 'fro')
        errs.append(err) # err 저장
        
        iter += 1
    S = Xs[-1]
    return {'sol':S, 'Xs':Xs, 'P_Xs':P_Xs, 'Hs':Hs, 'errs':errs}

def Pnomial(X, A):
#     S = np.zeros((X.shape[0],X.shape[1]))
#     for k in range(A.shape[0]):
#         S = S + A[k,:,:] @ nla.matrix_power(X, k)
    Y = A[-1,:,:]
    for i in range(2, A.shape[0]+1):
        Y = A[-i,:,:] + Y @ X # 계산 수를 줄인 polynomial 계산 함수
    return Y

def MakeStochA(dim, degree = 1):
    A = np.random.rand(degree + 1, dim, dim)
    b = np.sum(A, axis = (0, 2))
    for i in range(degree +1):
        for j in range(dim):
            A[i,j,:] = A[i,j,:]/b[j]
    r = np.max(np.abs(nla.eigvals(np.sum(A,0))))
    
    for i in range(dim):
        A[1,i,i] = A[1,i,i] - r
        
    return A
