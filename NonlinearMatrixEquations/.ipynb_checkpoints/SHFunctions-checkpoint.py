#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import linalg as nla
import scipy as sp
from scipy import linalg as sla
from time import time
from matplotlib import pyplot as plt
import os

def NewtonPoly(A, X0 = np.NAN, maxiter = 100, tol = np.NAN, cls = 'Pure', LS_iter = 2, alpha = .5):
    if np.sum(np.isnan(X0)) > 0: # X0가 주어지지 않았을 때 m by m zero 행렬 처리
        X0 = np.zeros((A.shape[1],A.shape[2]))
    
    if A.shape[1] != A.shape[2]: # A가 square matrices의 모음이 아닐 때 예외처리
        raise ValueError('A가 정방행렬이 아닙니다.')
        
    m, n = A.shape[1], A.shape[0]-1 # m, n 초기화
    
    if np.isnan(tol): # tol이 주어지지 않았을 때 초기화
        tol = m * 1e-15
        
    Xs = [X0] # Xs는 X들을 담은 리스트로 초기화
    P_Xs, Hs = [], [] # P_X, H 리스트 초기화
    S = np.zeros((A.shape[1],A.shape[2]))
    
    iter = 0
    err = nla.norm(Pnomial(X0, A), 'fro') # error 초기화
    errs = [err] # error 리스트 초기화
    L = []
    
    # Newton Iteration 시작
    c_time = time()
    while (err > tol) and (iter < maxiter):
        P_X = np.zeros((m**2, m**2))
        for k in range(1,n+1):
            for l in range(k):
                P_X = P_X + np.kron(nla.matrix_power(X0.transpose(),k-l-1), A[k,:,:] @ nla.matrix_power(X0, l))
        P_Xs.append(P_X) # P_X_i 저장
        vP = np.reshape(Pnomial(X0, A), m**2, 'F') # Reshape는 반드시 Fortran 방식으로
        h = nla.solve(P_X, vP)
#         h = nla.lstsq(P_X, vP)[0]
        H = np.reshape(h, (m, m), order = 'F')
        
        if cls == 'Pure':
            X0 = X0 - H # Newton Sequence 적용
            err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
            
            Xs.append(X0) # X_i 저장
            Hs.append(H) # H_i 저장
            errs.append(err) # err 저장
        
        elif cls == 'Kelley':
            if iter % 2 == 0:
                X0 = X0 - H # Newton Sequence 적용
            else:
                X0 = X0 - (2 - (nla.norm(H, 'fro')**alpha)) * H
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
            
        elif cls == 'MLSearch':
            if iter != LS_iter:
                X0 = X0 - H # Newton Sequence 적용
                err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
                L.append(1)
            else:
                pt = np.zeros(2*n + 1)
                for t in range(2*n + 1):
                    pt[t] = CoeffiLSearch(A, X0, -H, t)
                pt = np.flip(pt)
                pt = pt/np.min(np.abs(pt))
                ptder = np.polyder(pt)
                critic = np.roots(ptder)
                val = np.polyval(pt, critic)
                val = np.where(np.logical_and(critic >= 1, critic <= 2), val, np.infty)
                if np.sum(np.isinf(val)) == len(val):
                    lamb = 1
                else:
                    lamb = np.real(critic[np.argmin(np.abs(val))])
                L.append(lamb)
                
                X0 = X0 - lamb * H # Newton Line Search 적용
                err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
                
            Xs.append(X0) # X_i 저장
            Hs.append(H) # H_i 저장
            errs.append(err) # err 저장
            
        elif cls == 'ELSearch':
            pt = np.zeros(2*n + 1)
            for t in range(2*n + 1):
                pt[t] = CoeffiLSearch(A, X0, -H, t)
            pt = np.flip(pt)
            pt = pt/np.min(np.abs(pt))
            ptder = np.polyder(pt)
            critic = np.roots(ptder)
            val = np.polyval(pt, critic)
            val = np.where(np.logical_and(critic >= 1, critic <= 2), val, np.infty)
            if np.sum(np.isinf(val)) == len(val):
                lamb = 1
            else:
                lamb = np.real(critic[np.argmin(np.abs(val))])
            L.append(lamb)

            X0 = X0 - lamb * H # Newton Line Search 적용
            err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
                
            Xs.append(X0) # X_i 저장
            Hs.append(H) # H_i 저장
            errs.append(err) # err 저장
            
        elif cls == 'MLSearch2': # 새로운 ModifiedLineSearch
            if iter != LS_iter:
                X0 = X0 - H # Newton Sequence 적용
                err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
                L.append(1)
            else:
                n = A.shape[0]-1
                B = np.zeros(A.shape)
                for k in range(n+1):
                    for j in range(k+1):
                        B[j,:,:] = B[j,:,:] + A[k,:,:] @ PHIX(X0, -H, k, j)
                        
                p = []
                for i in range(A.shape[1]):
                    q = []
                    for j in range(A.shape[2]):
                        q.append(np.flip(B[:,i,j]))
                    p.append(q)
                
                rtss = [[None]*A.shape[2]]*A.shape[1]
                rts = [[None]*A.shape[2]]*A.shape[1]
                for i in range(A.shape[1]):
                    for j in range(A.shape[2]):
                        rtss[i][j] = np.roots(p[i][j])
                prts = np.min(np.real(np.where(np.logical_and(rtss[i][j] < 2, 1 <= rtss[i][j]), rtss[i][j], np.inf)))
                rts = [np.where(np.isinf(prts), 1, prts) for i in range(A.shape[1]) for j in range(A.shape[2])]
                lamb = np.min(rts)
                L.append(lamb)
                
                X0 = X0 - lamb * H # Newton Line Search 적용
                err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
                
            Xs.append(X0) # X_i 저장
            Hs.append(H) # H_i 저장
            errs.append(err) # err 저장
            
        elif cls == 'BLSInner': # Innerproduct를 이용하는 ModifiedLineSearch
            if iter != 0:
                cos = np.dot(np.reshape(H, m**2, order = 'F'), np.reshape(H0, m**2, order = 'F'))/(nla.norm(H)*nla.norm(H0))
            else:
                cos = 0
            
            if np.abs(cos - 1) > 1e-15:
                X0 = X0 - H # Newton Sequence 적용
                H0 = H
                err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
                L.append(1)
            else:
                n = A.shape[0]-1
                B = np.zeros(A.shape)
                for k in range(n+1):
                    for j in range(k+1):
                        B[j,:,:] = B[j,:,:] + A[k,:,:] @ PHIX(X0, -H, k, j)
                
                L1 = np.ones(A.shape[1:])
                L2 = 2 * L1
                Lc = (L1 + L2) / 2
                for i in range(52):    
#                     pc = np.asarray([[np.polyval(p[j][k], Lc[j,k]) for j in range(A.shape[1])] for k in range(A.shape[2])])
                    pc = ePnomial(Lc, B)
                    L1 = np.where(pc > 0, Lc, L1)
                    L2 = np.where(pc <= 0, Lc, L2)
                    Lc = (L1 + L2) / 2
                lamb = np.min(Lc)
                L.append(lamb)
                
                X0 = X0 - lamb * H # Newton Line Search 적용
                err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
            
            Xs.append(X0) # X_i 저장
            Hs.append(H) # H_i 저장
            errs.append(err) # err 저장
            
        elif cls == 'BLSearch': # BisectionLineSearch
            if iter != LS_iter:
                X0 = X0 - H # Newton Sequence 적용
                err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
                L.append(1)
            else:
                n = A.shape[0]-1
                B = np.zeros(A.shape)
                for k in range(n+1):
                    for j in range(k+1):
                        B[j,:,:] = B[j,:,:] + A[k,:,:] @ PHIX(X0, -H, k, j)
                
                L1 = np.ones(A.shape[1:])
                L2 = 2 * L1
                Lc = (L1 + L2) / 2
                for i in range(52):    
#                     pc = np.asarray([[np.polyval(p[j][k], Lc[j,k]) for j in range(A.shape[1])] for k in range(A.shape[2])])
                    pc = ePnomial(Lc, B)
                    L1 = np.where(pc > 0, Lc, L1)
                    L2 = np.where(pc <= 0, Lc, L2)
                    Lc = (L1 + L2) / 2
                lamb = np.min(Lc)
                L.append(lamb)
                
                X0 = X0 - lamb * H # Newton Line Search 적용
                err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
                
            Xs.append(X0) # X_i 저장
            Hs.append(H) # H_i 저장
            errs.append(err) # err 저장
            
        else:
            print('준비되지 않은 종류의 Newton method라 pure method로 전환합니다.')
            iter -= 1
            cls = 'Pure'
            
        iter += 1
    c_time = time() - c_time
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
    return {'sol':S, 'Xs':Xs, 'P_Xs':P_Xs, 'Hs':Hs, 'errs':errs, 'SmX':vSmX, 'XmX':vXmX, 'csSmX':cSX, 'csXmX':cXX, 'CalTime':c_time, 'lamb':L}

def CRPoly(A, X0 = np.NAN, maxiter = 100, tol = np.NAN):
    if np.sum(np.isnan(X0)) > 0: # X0가 주어지지 않았을 때 m by m zero 행렬 처리
        T0 = A[1]
        X0 = -nla.solve(A[1], A[0])
    
    if A.shape[1] != A.shape[2]: # A가 square matrices의 모음이 아닐 때 예외처리
        raise ValueError('A가 정방행렬이 아닙니다.')
        
    m, n = A.shape[1], A.shape[0]-1 # m, n 초기화
    
    if np.isnan(tol): # tol이 주어지지 않았을 때 초기화
        tol = m * 1e-15
        
    Xs = [X0] # Xs는 X들을 담은 리스트로 초기화
    # P_Xs, Hs = [], [] # P_X, H 리스트 초기화
    S = np.zeros((A.shape[1],A.shape[2]))
    
    iter = 0
    err = nla.norm(Pnomial(X0, A), 'fro') # error 초기화
    errs = [err] # error 리스트 초기화
    L = []
    
    A0, B0, C0 = A[2], A[1], A[0]
    
    # Newton Iteration 시작
    c_time = time()
    while (err > tol) and (iter < maxiter):
        T0 = T0 - A0 @ nla.inv(B0) @ C0
        A1 = A0 @ nla.inv(B0) @ A0
        B1 = B0 - A0 @ nla.inv(B0) @ C0 - C0 @ nla.inv(B0) @ A0
        C1 = C0 @ nla.inv(B0) @ C0
        
        A0, B0, C0 = A1, B1, C1
        X0 = -nla.inv(T0) @ A[0]
        
        err = nla.norm(Pnomial(X0, A), 'fro') # err 계산

        Xs.append(X0) # X_i 저장
        errs.append(err) # err 저장
        print('{:03d}'.format(iter), end = '\r')
        iter += 1
        
    c_time = time() - c_time
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
    return {'sol':S, 'Xs':Xs, 'errs':errs, 'SmX':vSmX, 'XmX':vXmX, 'csSmX':cSX, 'csXmX':cXX, 'CalTime':c_time}

def BrunoCRPoly(A, X0 = np.NAN, maxiter = 100, tol = np.NAN, criteria = 0):
    if np.sum(np.isnan(X0)) > 0: # X0가 주어지지 않았을 때 m by m zero 행렬 처리
        AH, AT = A[1], A[1]
        X0 = -nla.solve(A[1], A[0])
    
    if A.shape[1] != A.shape[2]: # A가 square matrices의 모음이 아닐 때 예외처리
        raise ValueError('A가 정방행렬이 아닙니다.')
        
    m, n = A.shape[1], A.shape[0]-1 # m, n 초기화
    
    if np.isnan(tol): # tol이 주어지지 않았을 때 초기화
        tol = m * 1e-15
        
    Xs = [X0] # Xs는 X들을 담은 리스트로 초기화
    S = np.zeros((A.shape[1],A.shape[2]))
    
    iter = 0
    err = nla.norm(Pnomial(X0, A), 'fro') # error 초기화
    errs = [err] # error 리스트 초기화
    L = []
    
    A0, B0, C0 = A[2], A[1], A[0]
    
    # Newton Iteration 시작
    c_time = time()
    while (err > tol) and (iter < maxiter):
        F = np.concatenate((C0, A0), axis=0) @ nla.inv(B0)
        F0 = F[:m, :]
        F2 = F[m:, :]        
        W = F2 @ C0
        C0 = F0 @ C0
        AH = AH - W
        B0 = B0 - W
        W = F0 @ A0
        A0 = F2 @ A0
        AT = AT - W
        B0 = B0 - W
        
        X0 = -nla.inv(AH) @ A[0]
        
        if criteria == 1:
            err = np.min([nla.norm(C0, 1), nla.norm(A0, 1)]) # err 계산
            errs.append(nla.norm(Pnomial(X0, A), 'fro')) # err 저장
        else:
            err = nla.norm(Pnomial(X0, A), 'fro') # err 계산
            errs.append(err) # err 저장
        
        Xs.append(X0) # X_i 저장
        print('{:03d}'.format(iter), end = '\r')
        iter += 1
        
    c_time = time() - c_time
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
    return {'sol':S, 'Xs':Xs, 'errs':errs, 'SmX':vSmX, 'XmX':vXmX, 'csSmX':cSX, 'csXmX':cXX, 'CalTime':c_time}

def CoeffiLSearch(A, X, H, jq):
    n = A.shape[0]-1
    S = np.zeros(X.shape)
    for p in range(n+1):
        for k in range(n+1):
            for q in range(p+1):
                for j in range(k+1):
                    if j+q == jq:
                        S = S + PHIX(X.transpose(), H.transpose(), p, q) @ A[p,:,:].transpose() @ A[k,:,:] @ PHIX(X, H, k, j)
    return np.trace(S)

def PHIX(X, H, k, j):
    if k == 0:
        S = np.eye(X.shape[0])
    else:
        L = []
        phi = [X, H]
        S = np.zeros(X.shape)
        P = np.eye(X.shape[0])
        for p in range(2**k):
            L.append(("{:{fill}"+str(k)+"b}").format(p, fill="0"))
        L = [[int(l) for l in ll] for ll in L]
        for l in L:
            if sum(l) == j:
                for i in l:
                    P = P @ phi[i]
                S = S + P
                P = np.eye(X.shape[0])
    return S

def Pnomial(X, A):
#     S = np.zeros((X.shape[0],X.shape[1]))
#     for k in range(A.shape[0]):
#         S = S + A[k,:,:] @ nla.matrix_power(X, k)
    Y = A[-1,:,:]
    for i in range(2, A.shape[0]+1):
        Y = A[-i,:,:] + Y @ X # 계산 수를 줄인 polynomial 계산 함수
    return Y

def ePnomial(X, A):
#     S = np.zeros((X.shape[0],X.shape[1]))
#     for k in range(A.shape[0]):
#         S = S + A[k,:,:] @ nla.matrix_power(X, k)
    Y = A[-1,:,:]
    for i in range(2, A.shape[0]+1):
        Y = A[-i,:,:] + Y * X # 계산 수를 줄인 polynomial 계산 함수
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

def MakeGenSingA(dim, degree, delta = 0, ZDiag = True):
    A = np.zeros((degree+1, dim, dim))
    
    if degree == 2:
        A[0,:,:] = genW(dim, delta, ZDiag)
        A[1,:,:] = genW(dim, delta, ZDiag) - 3*np.eye(dim)
        A[2,:,:] = genW(dim, delta, ZDiag) + 3*delta * np.eye(dim)
        A /= 3
    elif degree == 4:
        A[0,:,:] = 16*genW(dim, delta, ZDiag)
        A[1,:,:] = 2*genW(dim, delta, ZDiag) - 26*np.eye(dim)
        A[2,:,:] = genW(dim, delta, ZDiag)
        A[3,:,:] = 6*genW(dim, delta, ZDiag)
        A[4,:,:] = genW(dim, delta, ZDiag) + 26*delta * np.eye(dim)
        A /= 26
    elif degree == 6:
        A[0,:,:] = 4096*genW(dim, delta, ZDiag)
        A[1,:,:] = 56*genW(dim, delta, ZDiag) - 6200*np.eye(dim)
        A[2,:,:] = 384*genW(dim, delta, ZDiag)
        A[3,:,:] = 1312*genW(dim, delta, ZDiag)
        A[4,:,:] = 321*genW(dim, delta, ZDiag)
        A[5,:,:] = 30*genW(dim, delta, ZDiag)
        A[6,:,:] = genW(dim, delta, ZDiag) + 6200*delta * np.eye(dim)
        A /= 6200
    elif degree == 8:
        A[0,:,:] = 2985984*genW(dim, delta, ZDiag)
        A[1,:,:] = 21024*genW(dim, delta, ZDiag) - 4500000*np.eye(dim)
        A[2,:,:] = 311040*genW(dim, delta, ZDiag)
        A[3,:,:] = 905472*genW(dim, delta, ZDiag)
        A[4,:,:] = 244080*genW(dim, delta, ZDiag)
        A[5,:,:] = 30312*genW(dim, delta, ZDiag)
        A[6,:,:] = 2017*genW(dim, delta, ZDiag)
        A[7,:,:] = 70*genW(dim, delta, ZDiag)
        A[8,:,:] = genW(dim, delta, ZDiag) + 4500000*delta*np.eye(dim)
        A /= 4500000
    else:
        raise ValueError('degree 2, 4, 6, 8 외에는 준비되지 않았습니다.')
    return A

def MakeOneSingA(dim, degree, delta = 0, ZDiag = True):
    A = np.zeros((degree+1, dim, dim))
    W = (1 - delta) * (np.ones((dim, dim)) - np.eye(dim)) / (dim - 1)
    P = np.zeros((dim,dim))
    P[:,0] = np.ones(dim)
    for i in range(1,dim):
        P[i,i] = -1
        P[i-1,i] = 1
    S = 0
    
    if degree == 2:
        A[0,:,:] = W
        A[1,:,:] = W - 3*np.eye(dim)
        A[2,:,:] = W + 3*delta * np.eye(dim)
        A /= 3
        
        s1 = (2 - 2*delta) / (2 + 4*delta)
        s2s = np.roots([(3*dim-2)*delta - 1, delta + 2 - 3*dim, delta - 1])
        s2a = np.abs(s2s)
        s2 = s2s[np.argmin(s2a)]
        D = s2 * np.ones(dim)
        D[0] = s1
        S = P @ np.diag(D) @ nla.inv(P)
        
    elif degree == 4:
        A[0,:,:] = 16*W
        A[1,:,:] = 2*W - 26*np.eye(dim)
        A[2,:,:] = W
        A[3,:,:] = 6*W
        A[4,:,:] = W + 26*delta * np.eye(dim)
        A /= 26
        
        s1s = np.roots([1+25*delta, 6-6*delta, 1-delta, -24-2*delta, 16-16*delta])
        s2s = np.roots([26*dim*delta - 25*delta - 1, 6*delta - 6, delta - 1, 24 + 2*delta - 26*dim, 16*delta - 16])
        s1a = np.abs(s1s)
        s2a = np.abs(s2s)
        s1 = s1s[np.argmin(s1a)]
        s2 = s2s[np.argmin(s2a)]
        D = s2 * np.ones(dim)
        D[0] = s1
        S = P @ np.diag(D) @ nla.inv(P)
        
    elif degree == 6:
        A[0,:,:] = 4096*W
        A[1,:,:] = 56*W - 6200*np.eye(dim)
        A[2,:,:] = 384*W
        A[3,:,:] = 1312*W
        A[4,:,:] = 321*W
        A[5,:,:] = 30*W
        A[6,:,:] = W + 6200*delta * np.eye(dim)
        A /= 6200
        
        s1s = np.roots([1+6199*delta, 30*(1-delta), 321*(1-delta), 1312*(1-delta), 384*(1-delta), -6144 - 56*delta, 4096*(1-delta)])
        s2s = np.roots([6200*dim*delta - 1 - 6199*delta, 30*(delta-1), 321*(delta-1), 1312*(delta-1), 384*(delta-1), 56*delta + 6144 - 6200*dim, 4096*(delta-1)])
        s1a = np.abs(s1s)
        s2a = np.abs(s2s)
        s1 = s1s[np.argmin(s1a)]
        s2 = s2s[np.argmin(s2a)]
        D = s2 * np.ones(dim)
        D[0] = s1
        S = P @ np.diag(D) @ nla.inv(P)
        
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
        
        s1s = np.roots([1+4499999*delta, 70*(1-delta), 2017*(1-delta), 30312*(1-delta), 244080*(1-delta), 905472*(1-delta), 311040*(1-delta), -21024*delta - 4478976, 2985984*(1-delta)])
        s2s = np.roots([4500000*dim*delta - 1 - 4499999*delta, 70*(delta-1), 2017*(delta-1), 30312*(delta-1), 244080*(delta-1), 905472*(delta-1), 311040*(delta-1), 4478976 + 21024*delta - 4500000*dim, 2985984*(delta-1)])
        s1a = np.abs(s1s)
        s2a = np.abs(s2s)
        s1 = s1s[np.argmin(s1a)]
        s2 = s2s[np.argmin(s2a)]
        D = s2 * np.ones(dim)
        D[0] = s1
        S = P @ np.diag(D) @ nla.inv(P)
        
    else:
        raise ValueError('degree 2, 4, 6, 8 외에는 준비되지 않았습니다.')
    return A, S

def genW(dim, delta = 0, ZDiag = True):
    K = np.random.rand(dim, dim)
    if ZDiag == True:
        for i in range(dim):
            K[i,i] = 0
    k = np.sum(K, axis = 1)
    for j in range(dim):
        K[j,:] = K[j,:]/k[j]
    W = (1 - delta) * K
    return W
