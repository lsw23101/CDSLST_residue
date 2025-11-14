import numpy as np
from scipy.io import loadmat
from sympy import isprime
from enc_func import *

# 이 파일이 해야하는 일:
# F_bar G_bar H_bar 보내주고
# 60세트의 (T1 V1 T2 V2 / S_xi S_v / Psi, Sigma, Sigma_pinv) 보내주기
# 

# 여기까지 유틸함수

env = Params()

def main():
    # 1) MATLAB mat 파일 로드
    data = loadmat('FGH_data.mat')
    F_ = data['F_bar']   # float64
    G_ = data['G_']      # float64
    H_ = data['H']       # float64

    # 2) 양자화 및 정수 변환
    s = 10000

    F_bar_float = F_                   # F_bar는 원래 정수라면 그대로
    G_bar_float = np.rint(s * G_)
    H_bar_float = np.rint(s * H_)

    F_bar = np.vectorize(int)(F_bar_float)   # dtype=object, elements = int
    G_bar = np.vectorize(int)(G_bar_float)
    H_bar = np.vectorize(int)(H_bar_float)

    # 3) H_bar의 1행 추출
    H1 = H_bar[0, :].copy()

    print("H1 =", H1)


    # 2) 함수 호출
    T1, T2, T, V, V1, V2 = build_TV(H1, env.q)

    # 18번 밑의 행렬 값 정리 / S_1 S_2 S_3 psi gamma sigma  6개 + Sig_pinv 1개

    S_1  = Mod(T1 @ F_bar @ V1, env.q)   # 23x23
    S_2  = Mod(T1 @ F_bar @ V2, env.q)   # 23x1
    S_3  = Mod(T1 @ G_bar,      env.q)   # 23x6

    Psi  = Mod(H1.reshape(1,24) @ F_bar @ V1, env.q)  # 1x23
    Gamma = Mod(H1.reshape(1,24) @ F_bar @ V2, env.q)  # 1x1
    Sigma = Mod(H1.reshape(1,24) @ G_bar,      env.q)  # 1x6
    
    sigma0 = int(Sigma[0, 0])
    if sigma0 == 0:
        print("\n[경고] Sigma[0,0] ≡ 0 (mod q) 이라서 첫 원소로는 right inverse를 만들 수 없습니다.")
    else:
        inv_sigma0 = pow(sigma0, -1, env.q)  
        Sigma_pinv = np.zeros((6, 1), dtype=object)
        Sigma_pinv[0, 0] = inv_sigma0


    # xi 의 동역학 행렬 / S_xi 23x23  S_v 23x6
    # b_xi+ = S_xi b_xi + S_v b_v 로 상태변수 업데이트 될 예정

    S_xi = Mod(S_1 - S_3 @ Sigma_pinv @ Psi, env.q)
    S_v = Mod(S_3 @ (np.zeros((6, 6), dtype=object) - Sigma_pinv@Sigma),env.q)

    print("S_xi, S_v", S_xi.shape, S_v.shape) 


    # Sigma_pinv @ b_tilde = b_prime 계산하는 함수 
    # input = b_v : 6x1 행렬, state = b_xi 
    
    # b_v = [1 1 1 1 1 1]  # 6x1 그냥 인풋 예시
    # b_xi = [1 1 1 ...]   # 23x1 state 예시
    # b_prime = Sigma_pinv @ ( Sigma @ b_v + Psi @ b_xi) 

    # # 3) 결과 출력
    # print("\nT1 ="); print(T1)
    # print("\nT2 ="); print(T2)
    # print("\nV1 ="); print(V1)
    # print("\nV2 ="); print(V2)

    # # 4) 검증 연산
    # print("\n=== V * T (mod q) ===")
    # print(matmul_mod(V, T, env.q))

    # print("\n=== T * V (mod q) ===")
    # print(matmul_mod(T, V, env.q))

    # print("\n=== T1 * V1 (mod q) ===")
    # print(matmul_mod(T1, V1, env.q))

    # print("\n=== T2 * V2 (mod q) ===")
    # res = matmul_mod(T2, V2, env.q)
    # print(res, "  (scalar:", res[0, 0], ")")
    

if __name__ == "__main__":
    main()