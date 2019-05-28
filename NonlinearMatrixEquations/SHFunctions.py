#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import linalg as nla
import scipy as sp
from scipy import linalg as sla
from time import time
from matplotlib import pyplot as plt
import os

def NewtonPoly(A, X0 = np.NAN, maxiter = 100, tol = np.NAN, cls = 'Pure'):
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
        
        if cls == 'Pure':
            X0 = X0 - H # Newton Sequence 적용
            err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
            
            Xs.append(X0) # X_i 저장
            Hs.append(H) # H_i 저장
            errs.append(err) # err 저장
        
        elif cls == 'Modified':
            X0 = X0 - 2*H # modified Newton Sequence 적용
            err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
            if err <= tol:
                Xs.append(X0) # X_i 저장
                Hs.append(H) # H_i 저장
                errs.append(err) # err 저장
                break
            X0 = X0 + H # pure Newton 재적용
            err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
            
            Xs.append(X0) # X_i 저장
            Hs.append(H) # H_i 저장
            errs.append(err) # err 저장
            
        else:
            print('준비되지 않은 종류의 Newton method라 pure method로 전환합니다.')
            cls = 'Pure'
            
        iter += 1
    
    S = Xs[-1] # Solution
    
    # Vectorize of S - X_{i}와 X_{i+1} - X_{i} : cos 계산
    vSmX = []
    vXmX = []
    for i in range(len(Xs)-1):
        vSmX.append(np.reshape(S - Xs[i], S.shape[0]*S.shape[1], order='F'))
        vXmX.append(np.reshape(Xs[i+1] - Xs[i], S.shape[0]*S.shape[1], order='F'))
    cSX = []
    cXX = []
    for i in range(len(vSmX)-1):
        x1, y1 = vSmX[i+1], vSmX[i]
        x2, y2 = vXmX[i+1], vXmX[i]
        c1 = np.dot(x1,y1) / (nla.norm(x1,2)*nla.norm(y1,2))
        c2 = np.dot(x2,y2) / (nla.norm(x2,2)*nla.norm(y2,2))
        cSX.append(c1)
        cXX.append(c2)
    return {'sol':S, 'Xs':Xs, 'P_Xs':P_Xs, 'Hs':Hs, 'errs':errs, 'SmX':vSmX, 'XmX':vXmX, 'csSmX':cSX, 'csXmX':cXX}

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

def MakeSingularA(dim, degree, delta = 0):
    A = np.zeros((degree+1, dim, dim))
    K = np.random.rand(dim, dim)
    for i in range(dim):
        K[i,i] = 0
    k = np.sum(K, axis = 1)
    for j in range(dim):
        K[j,:] = K[j,:]/k[j]
    W = (1 - delta) * K
    
    if degree == 2:
        A[0,:,:] = W
        A[1,:,:] = W - 3*np.eye(dim)
        A[2,:,:] = W + 3*delta * np.eye(dim)
        A /= 3
    elif degree == 4:
        A[0,:,:] = 16*W
        A[1,:,:] = 2*W - 26*np.eye(dim)
        A[2,:,:] = W
        A[3,:,:] = 6*W
        A[4,:,:] = W + 26*delta * np.eye(dim)
        A /= 26
    elif degree == 6:
        A[0,:,:] = 4096*W
        A[1,:,:] = 56*W - 6200*np.eye(dim)
        A[2,:,:] = 384*W
        A[3,:,:] = 1312*W
        A[4,:,:] = 321*W
        A[5,:,:] = 30*W
        A[6,:,:] = W + 6200*delta * np.eye(dim)
        A /= 6200
    elif degree == 8:
        A[0,:,:] = 2985984*W
        A[1,:,:] = 21024*W - 4500000*np.eye(dim)
        A[2,:,:] = 311040*W
        A[3,:,:] = 905472*W
        A[4,:,:] = 244080*W
        A[5,:,:] = 30312*W
        A[6,:,:] = 2017*W
        A[7,:,:] = 70*W
        A[8,:,:] = W + 4500000*delta*np.eye(dim)
        A /= 4500000
    else:
        raise ValueError('degree 2, 4, 6, 8 외에는 준비되지 않았습니다.')
    return A