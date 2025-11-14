import numpy as np
import random


# 이 파일이 해야하는 일:
# 비밀키 만들기
# 모듈러, 암호화 (초기값, 출력), 복호화 (24길이 혹은 길이 1 짜리 둘 다 호환 되도록)


class Params:
    def __init__(self):
        # 여기서 K의 최댓값은 355,700,000 (참고용)
        self.p = int(2**54)   # p 
        self.L = int(2**10)   # L 
        self.r = 10           # 오류 범위
        self.N = 4            # 키 차원 
        # 2^64 근처 소수
        self.q = self.p * self.L - 59    # 18446744073709551557

env = Params()


def Seret_key(env):
    # sk를 -1, 0, 1 중 하나로 무작위로 선택
    return np.array([[random.choice([-1, 0, 1])] for _ in range(env.N)], dtype=object)


def Mod(x, p):
    """
    Centered modular:
    x (스칼라 또는 배열)을 정수로 보고 mod p 한 뒤,
    결과를 [-p/2, p/2] 범위로 옮겨줌.
    모든 연산은 파이썬 int 기반.
    """
    x_arr = np.asarray(x, dtype=object)

    def centered(v):
        v_int = int(v)
        r = v_int % p
        if r >= p // 2:
            r -= p
        return r

    return np.vectorize(centered, otypes=[object])(x_arr)

"""
## Enc_state (z_hat_quantized, sk, env, T1, T2, V2)
#  A = 랜덤
#  e = 랜덤
#  b_ini = A sk + e \in 24 
#  b_tilde_ini = T2 @ b_ini \in 1  << b_ini에 T2 행벡터 내적한 값
#  b_xi_ini = T1 @ b_ini    \in 23 << 사실상 b_ini의 아래 23개
#  b_ini_prime = V2 @ b_tilde_ini   << 내적한값에 24x1로 임베딩
#  
#  출력 : [z_bar_ini + b_ini - b_ini_prime, A, b_ini_prime], b_xi_ini
## 
"""
# 여기서 b_xi_ini 뽑아서 꺼내놓을 예정 
# 그리고 이 b_xi_ini가 60개 돌아갈 예정


# 위에서 꺼낸거를 이제 밑에 다이나믹 암호화
"""
## Enc_t (v, sk, b_xi, S_xi, S_v, Sigma_pinv, Sigma_pinv, Psi, env) 를 받아서 (S_xi, S_v는 미리 계산, b_xi는 루프마다 업데이트 될 예정)
#   
#   Av = 랜덤하게 뽑기
#   e = 랜덤하게 뽑기
#   bv = sk Av + e 
#
#   ## b_xi = S_xi @ b_xi + S_v @ b_v  # 업데이트 되는 요소 (함수 밖에서 하고 줄 예정, 시뮬레이션을 위해)
#
#   b_prime =  Sigma_pinv @ ( Sigma @ b_v + Psi @ b_xi) 
#   
## 출력 : [v + bv - b_prime , Av, b_prime] 
##
"""

"""
## Dec(암호문, sk, env) 
#  암호문 : h x N+2 
#  m_bar = Mod(암호문 @ [1 -sk 1]을 세로로 세우고 h개 만큼 가로로 늘린 행렬: N+2 x h) 
#  
#  m = b_nar / L
# 
##  출력 : [m] \in h
#

"""


# 이거 안쓸거같은데
def matmul_mod(A, B, q):
    """Z_q 위에서의 행렬 곱: C = A @ B (mod q)"""
    n, m = A.shape
    m2, k = B.shape
    assert m == m2
    C = np.zeros((n, k), dtype=object)
    for i in range(n):
        for j in range(k):
            s = 0
            for l in range(m):
                s += int(A[i, l]) * int(B[l, j])
            C[i, j] = s % q
    return C


def build_TV(H1, q):
    """
    입력:  
        H1 : 1×24 numpy object vector  
        q  : modulus

    출력:  
        T1, T2, T, V, V1, V2
    """
    # --- 1) T1 생성 (23×24, [0 | I]) ---
    T1 = np.zeros((23, 24), dtype=object)
    for i in range(23):
        T1[i, i+1] = 1

    # --- 2) T2 (= H1) ---
    T2 = H1.reshape(1, 24)

    # --- 3) T = [T1; T2] ---
    T = np.vstack([T1, T2])

    # --- 4) V 계산 ---
    h0 = int(H1[0]) % q
    inv_h0 = pow(h0, -1, q)

    V = np.zeros((24, 24), dtype=object)

    # 아래 23×23 블록 항등행렬
    for i in range(23):
        V[i+1, i] = 1

    # 1행 채우기 (mod q)
    for j in range(23):
        hj = int(H1[j+1]) % q
        V[0, j] = (-hj * inv_h0) % q

    V[0, 23] = inv_h0 % q

    # --- 5) V1, V2 ---
    V1 = V[:, :23].copy()
    V2 = V[:, 23].reshape(24, 1)

    return T1, T2, T, V, V1, V2