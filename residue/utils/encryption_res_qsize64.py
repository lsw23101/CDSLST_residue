# J의 q에 대한 역원 4352491850465363121987

import numpy as np
import random
from decimal import Decimal

# from sympy import mod_inverse

# 환경 변수 설정
class params:
    def __init__(self):
        # 여기서 K의 최댓값은 355,700,000
        self.p = int(2**54)  # p 
        self.L = int(2**10)  # L 
        self.r = 10         # 오류 범위
        self.N = 4     # 키 차원 
        self.q = self.p * self.L -59 # 2^64 근처 소수 18446744073709551557
        
env = params()

def Mod(x, p):
    y = np.mod(x, p)  # 요소별 모듈러 계산
    y -= (y >= p / 2) * p  # 범위 조정 (biased modular)
    return y


def Seret_key(env):
    # sk를 -1, 0, 1 중 하나로 무작위로 선택
    return np.array([[random.choice([-1, 0, 1])] for _ in range(env.N)], dtype=object)

sk = Seret_key(env) 
# print("sk", sk)

# 암호화 함수

def Enc_state(m_vec, sk, env):
    n = len(m_vec)  # 메시지 벡터 크기 
    
    # 난수 행렬 A 생성
    A = np.array([[random.randint(0, env.q - 1) for _ in range(env.N)] for _ in range(n)], dtype=object)
    A = Mod(A, env.q)  # 모듈러 연산 적용
    
    e = np.random.normal(0, env.r, size=(n, 1)).astype(int)  
    
    mask = Mod(A @ sk + e,env.q) 

    m_vec = np.round(m_vec)  # 벡터에 대해 반올림 처리
    
    b = Mod(mask + env.L * m_vec, env.q)  # 마스크된 메시지

    # print("b",b)
    
    # 암호문 조합
    ciphertext = Mod(np.hstack((b, A, np.zeros((n, 1), dtype=object))), env.q)
    # print("ciphertext", ciphertext)
    return ciphertext, mask

# 암호화 함수
def Enc_res(m, sk, Bx, M, env):
    n = 2  # 메시지는 y 가 2차원

    # `np.random.randint()` 대신 Python `random.randint()` 사용하여 A 생성
    A = np.array([[random.randint(0, env.q - 1) for _ in range(env.N)] for _ in range(n)], dtype=object)
    A = Mod(A, env.q)  # 모듈러 연산 적용

    e = np.random.normal(0, env.r, size=(n, 1)).astype(int)  # 가우시안 분포에서 오류 샘플링

    # m = np.round(m).astype(object)  # 메시지가 float일 경우 반올림 처리
    
    mask = M @ Bx

    k = Mod(A @ sk + e + mask, env.q)  # 마스크된 메시지

    # 암호문 조합
    ciphertext = Mod(np.hstack((-mask + env.L*m , A, k)), env.q)
    return ciphertext


def Dec_res(c, sk, env):
    s = np.vstack((1, -sk, 1))  # 키 벡터

    decrypted = Mod(c @ s, env.q)  # 암호문 연산 후 모듈러 연산 추가

    # Decimal 사용하여 안전한 나눗셈 수행
    decrypt_decimal = (decrypted // env.L).astype(object)  # object 타입으로 나눔
    
    # 소수점 반올림 후 정수 변환
    plaintext = np.array([round(d) for d in decrypt_decimal.flatten()], dtype=object)

    return plaintext.reshape(decrypted.shape)  # 원래 모양 유지