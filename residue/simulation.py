# import 할것들
import numpy as np
from scipy.io import loadmat
from sympy import isprime
from enc_func import *



# 파라미터 생성
Ts = 1 # 샘플링타임 1초
env = Params()  # 환경 설정
sk = Seret_key(env)
# print("q is prime?", isprime(env.q))
# print("N is", env.N)


# 오프라인 행렬들 준비


# 여기서 준비가 끝날 것들:
# 이산화된 A B C 행렬

A = np.array([
    [ 0.572915, 0.222492, 0.294165, 0.228264, 0.132920, 0.268409 ],
    [ 0.790746, 0.417171, 4.709138, 0.134381, -5.499884, -0.054966 ],
    [ 0.294165, 0.228264, 0.411670, 0.262637, 0.294165, 0.228264 ],
    [ 4.709138, 0.134381, -9.418275, 0.227824, 4.709138, 0.134381 ],
    [ 0.132920, 0.268409, 0.294165, 0.228264, 0.572915, 0.222492 ],
    [ -5.499884, -0.054966, 4.709138, 0.134381, 0.790746, 0.417171 ]
], dtype=np.float64)

B = np.array([ 13.613318, 22.249166, 13.301577, 22.826353, 13.204555, 26.840867 ], dtype=np.float64)  # shape (6,)

C = np.array([
    [ 1, 0, 0, 0, 0, 0 ],
    [ 0, 0, 1, 0, 0, 0 ],
    [ 0, 0, 0, 0, 1, 0 ],
    [ 1, 0, -1, 0, 0, 0 ],
    [ 0, 0, 1, 0, -1, 0 ]
], dtype=np.int64)


# K 피드백 게인
K = np.array([ -0.010617, -0.010772, -0.009818, -0.010584, -0.007183, -0.011472 ], dtype=np.float64)  # shape (6,)

# F_bar G_bar H_bar 
# T1 T2 V1 V2  : 60세트
# S_xi, S_v  (b_xi의 상태공간 행렬) : 60세트
# 

# 위 행렬 3개 + 60 * 6개 행렬들은 offline_task.py 에서 불러오기



# 초기값 설정
# 데이터 저장할 배열 설정

iter = 240
execution_times = []  # 실행 시간을 저장할 리스트

# 양자화 파라미터

r_quant = 10000
s_quant = 10000



# 초기값

xp0 = np.array([[0.1], [0.1], [0.1], [0.1]])
z_hat0 = np.array([[0], [0], [0], [0]])
# Initialize variables with proper int conversion at the beginning

# 실수 위에서의 초기값
xp, z_hat, u, y = [xp0], [z_hat0], [], []
z_hat_bar = np.round(z_hat * r_quant * s_quant).astype(int)




## Enc_state (z_hat_quantized, sk, env)
#  A = 랜덤
#  e = 랜덤
#  b_ini = A sk + e \in 24 
#  b_tilde_ini = T2 @ b_ini \in 1  << b_ini에 T2 행벡터 내적한 값
#  b_xi_ini = T1 @ b_ini    \in 23 << 사실상 b_ini의 아래 23개
#  b_ini_prime = V2 @ b_tilde_ini   << 내적한값에 24x1로 임베딩
#  
#  출력 : [z_bar_ini + b_ini - b_ini_prime, A, b_ini_prime], b_xi_ini
## 

# 여기서 b_xi_ini 뽑아서 꺼내놓을 예정 
# 그리고 이 b_xi_ini가 60개 돌아갈 예정


# 위에서 꺼낸거를 이제 밑에 다이나믹 암호화

## Enc_t (v, sk, b_xi, S_xi, S_v, env) 를 받아서 (S_xi, S_v는 미리 계산, b_xi는 루프마다 업데이트 될 예정)
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


# 암호화 된 z_hat 초기 값 
# 
# 암호문은 대문자로 쓰자고 하자
# Z_hat도 60개
Z_hat, b_xi_ini = Enc_state(z_hat_bar, sk, env)  # 여긱서 암호화 할때 cX0 의 마스킹 파트도 뽑아냄

# 여기서 b_xi_ini 는 60개

# Simulation loop
for i in range(iter):
    # 플랜트 출력 
    y.append(C @ x_p[-1])

    # 피드백 제어 입력 
    u.append(K @ x_p)  

    # 옵저버 출력
    # 옵저버 연산 j=1 부터 60까지 
    # 식 (27)에서 R_bar = H_bar @ Z_hat
    # 이거를 60번 R_bar_j \in 1 x N+2
    
    # 첫번째 값만 꺼내고 양자화 돌려놓기 
    
    # 결과 60개를 다시 쌓아서 r_bar 만듦
    # r_bar \in 60 x 1
    # 리스트에 추가
    r_bar.append(lwe.Mod(H_bar @ r_bar, env.q))
    


    ###  출력 암호화
    # v=[u;y] 쌓고 양자화 및 암호화 j=1 부터 60까지 
    # 암호화는 b_tilde를 받아서 암호화
    
    # 양자화
    v_bar = np.round(r_bar[-1] * r_quant)

    # 암호화 
    V = Enc_t (v_bar, sk, b_xi, S_xi, S_v, env)

    ###


    #### 출력으로 얻은 r_bar 를 양자화 맞춰서 실수로 만든 r \in R^60
    #
    # 여기서 공격 당하지 않은 람다를 찾을 수 있음
    # 그 람다에 해당하는 Enc(z) 를 복호화 하면 state 추정 가능
    # 이 r 값이 식(13)의 좌변
    # 실제 상태 xp와 
    # 인덱스하나 고른거에서의 Z_Lamda를 복호화 하여 
    # x_bar_Lambda 를 복호화하여 x_lamda 값을 비교하는것이 Thm3의 2번 식
    ####


    ### 동적 업데이트 부분 ###

    # 플랜트 상태 업데이트
    xp.append(A @ xp[-1] + B @ u[-1])  

    # 암호화 된 옵저버 상태 업데이트
    Z_hat = Mod(F_bar @ Z_hat + G_bar @ V , env.q) 

    # 마스킹 파트 업데이트
    # j=1 부터 60까지 
    # 플랜트단에서 마스킹파트를 동적 연산하는 부분 j=1 부터 60까지 
    b_xi = S_xi @ b_xi + S_v @ b_v 


    # 결과 정리 및 시간 측정 등등
    # 플랏할거 
    # 1. 공격신호 2. 길이 60 r 신호와 쓰레스홀드 비교 3. 상태추정이 발산하는것 즉 실제 상태와 추정 상태의 차이 
    ########### TO DEBUG ################