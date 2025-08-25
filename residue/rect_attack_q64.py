import numpy as np
import matplotlib.pyplot as plt
import utils.encryption_res_qsize64 as lwe  # 기존 암호화 함수는 lwe로 유지
import time
import random
from decimal import Decimal
from sympy import mod_inverse
from sympy import isprime
import datetime

np.seterr(over='raise', invalid='raise')  # 오버플로우 및 NaN 발생 시 에러 발생

# 파라미터 생성 // q L e 
Ts = 0.05 # 루프타임이 28ms 니까 50ms 샘플링타임으로 설정
env = lwe.params()  # 환경 설정
sk = lwe.Seret_key(env)
print("q is prime?", isprime(env.q))
print("N is", env.N)

# lattice-estimator로 보안레벨 출력
# import sys
# sys.path.append('./lattice-estimator')  # 현재 residue 폴더 기준

# try:
#     from lattice_estimator.estimator import *

#     n = int(env.N)
#     q = int(env.q)
#     r = int(env.r)

#     params = LWE.Parameters(
#         n=n,
#         q=q,
#         Xs=ND.Uniform(-1, 1, n=n),      # sk: -1, 0, 1
#         Xe=ND.Uniform(-r, r)            # error: -r ~ r
#     )

#     results = LWE.estimate.rough(params)
#     print('\nLWE lattice security estimate:')
#     print('------------------------------')
#     for attack, res in results.items():
#         print(f"{attack}: rop={res.get('rop', 'N/A')}")
#     print('full result:', results)
# except Exception as e:
#     print('lattice-estimator 실행 중 오류:', e)

############ Discretized Plant Model wiht 50ms #################
# A = np.array([[1.000000000000000, 0.009990914092165, 0.000133590122189, 0.000000445321570], 
#           [0.000000000000000, 0.998183267746166, 0.026716874509824, 0.000133590122189], 
#           [0.000000000000000, -0.000022719408536, 1.001559293628154, 0.010005197273892], 
#           [0.000000000000000, -0.004543686141126, 0.311919519484923, 1.001559293628154]])

# B = np.array([[0.000090859078355], 
#           [0.018167322538343], 
#           [0.000227194085355], 
#           [0.045436861411265]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

############ Controller Model with Re-Enc #################


# F_ should be integer
F_ = np.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])

J_ = np.array([[1, 0],
               [0, 1]])

'''
매트랩 복붙
'''
A = np.array([[1.000000000000000, 0.049773098368562, 0.003352519883901, 0.000055772534794], 
          [0.000000000000000, 0.990924994598992, 0.134769006497267, 0.003352519883901], 
          [0.000000000000000, -0.000570156442840, 1.039205686203506, 0.050651840518290], 
          [0.000000000000000, -0.022919899064161, 1.577892608941279, 1.039205686203506]])

B = np.array([[0.002269016314383], 
          [0.090750054010083], 
          [0.005701564428404], 
          [0.229198990641610]])


R_ = np.array([[-2.711203086644444, 1.963316224315184], 
          [-0.216982902179624, -0.313486996745260], 
          [-0.115423089429835, 0.730498316552361], 
          [0.055991142534054, -0.314052234754461]])

G_ = np.array([[2.045469069819882, 1.633066593297948], 
          [1.980615785336393, -1.866844647059324], 
          [-2.570268231442944, 0.757301003832312], 
          [-0.330294262457960, 0.244716632050707]])

H_ = np.array([[-0.461010620513824, -0.760867797057168, -0.000000000000000, 0.000000000000001], 
          [-0.407144784667603, -0.150389370126363, 0.000000000000000, 0.000000000000000]])

P_ = np.array([[-11.343158917072174, -6.626687755976155, -16.429184877793354, 116.215355988387330]])

# L*r*s*s ~ 2^48 정도
# 40배 정도 더 늘어난거니까

# Quantization parameters
r_scale = 10000
s_scale = 10000

scaled_value = s_scale**2

print("sclaed value", scaled_value)

qG = np.round(G_ * s_scale).astype(int)
qH = np.round(H_* s_scale).astype(int)
qP = np.round(P_ * s_scale).astype(int)
qJ = np.round(J_ * s_scale**2).astype(object)
qR = np.round(R_ * s_scale).astype(int)




############ Equivalent input with XC = [0 0 0.001 0]  ##########
## 

# To avoid small floating-point issues, let's scale 1/s^2 to an integer


# Now, compute the modular inverse of the scaled value
inverse = mod_inverse(scaled_value, env.q)

print("inverse:", inverse)

J_inv = (inverse * J_).astype(object) # GPT로 구한 2^64-59에서의 10^6의 inverse....
# J_inv = np.array([[2342055259102470954, 0],
#                 [0, 2342055259102470954]]).astype(int)

print(2342055259102470954*100000000)
print("J_inv @ qJ ", J_inv @ qJ)


M = np.vectorize(lambda x: int(lwe.Mod(x, env.q)))(J_inv @ qH)
W = np.vectorize(lambda x: int(lwe.Mod(x, env.q)))(F_ - qG @ J_inv @ qH)





print("M", M)
print("W", W)
print("inverse check", lwe.Mod(inverse*scaled_value, env.q))


print("qP:", qP)
print("qG:", qG)
print("qR:", qR)
print("qH:", qH)
print("qJ:", qJ)

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
# # ## ## ## ## ## ## ## ## Simulation settings # ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #

iter = 240
execution_times = []  # 실행 시간을 저장할 리스트

# 초기값

xp0 = np.array([[-0.1], [-0.1], [0.1], [0.1]])
xc0 = np.array([[0], [0], [0], [0]])
# Initialize variables with proper int conversion at the beginning

xp, xc, u, y = [xp0], [xc0], [], []
x_p, x_c, u_, y_, r_ = [xp0], [xc0], [], [], []
Xp, qXc, Xc, Y, U = [xp0], [np.round(xc0 * r_scale * s_scale).astype(int)], [xc0], [], []
resi, qY, qU, residue, cY, cU, cresi = [], [], [], [], [], [], []
diff_u, diff_Xc = [], []

qX0 = np.round(xc0 * r_scale * s_scale).astype(int)

# print("qX0: ",qX0)
# Encrypted controller initial state
cX0, Bx = lwe.Enc_state(qX0, sk, env)  # 여긱서 암호화 할때 cX0 의 마스킹 파트도 뽑아냄
# print("암호화 된 Xc",cX0)
# Simulation loop

disturbance_values = []

# 표준정규분포 외란 생성 (표준편차를 0.3으로 줄여서 0에 더 가까운 값들이 나오도록)
disturbance_sequence = np.random.normal(0, 0.5, iter) * 0.05

for i in range(iter):
    
    start_time = time.time()  # 시작 시간 기록 
    
    # 기본 표준정규분포 외란
    base_disturbance = disturbance_sequence[i]
    
    # 추가 impulse 어택 (200~500 이터레이션)
    additional_disturbance = 0
    if i > 200 and i < 500:
        additional_disturbance = 2
    
    # 총 외란 = 기본 표준정규분포 외란 + 추가 외란
    total_disturbance = base_disturbance + additional_disturbance
    
    disturbance_values.append(total_disturbance)  # disturbance 저장
    
    '''############# original 컨트롤러 ############## '''

    y_.append(C @ x_p[-1])
    u_.append(P_ @ x_c[-1] + total_disturbance)  # 표준정규분포 외란 사용
    r_.append(H_ @ x_c[-1] + J_ @ y_[-1])
    x_p.append(A @ x_p[-1] + B @ u_[-1])
    x_c.append(F_ @ x_c[-1] + G_ @ y_[-1] + R_ @ r_[-1])

    #########################################################
    #########################################################
    

    '''############# Encrypted controller ############## '''


    ###############################################
    #################### Plant ####################
    ###############################################

    # sensor

    Y.append(C @ Xp[-1])  # Y에 스칼라 값 저장
    qY.append(np.vectorize(lambda x: int(round(Decimal(x))), otypes=[object])(Y[-1] * r_scale))
    cY.append(lwe.Enc_res(qY[-1], sk, Bx, M,env))

    
    # print("cY", cY[-1])
    # controller
    
    cU.append(lwe.Mod(qP @ cX0, env.q))
    cU[-1][0][0] += total_disturbance * int(env.L * r_scale * s_scale**2)  # 표준정규분포 외란 사용
    # cU[-1][0][0] += disturbance * 10**15
    # print("cU",cU[-1])   # 첫 번째 요소에만 disturbance 더하기
    # print("cU[-1][0][0]",cU[-1][0][0])
    # 첫 번째 값에만 disturbance를 더하기
  

    ## 여기서 cresi가 첫 이터레이션때는 1이 나와야 됨
    cresi.append(lwe.Mod(qH @ cX0 + qJ @ cY[-1], env.q))  # encrypted controller output  

    # print("cresi[-1]", cresi[-1])
    # 여기서 나오는 cresi는 이론상 1/sr 로 스케일링 된 마스킹 파트가 없는 값
    # cresi [1/sr resi , A, B]

    # actuator
    qU.append(lwe.Dec_res(cU[-1], sk, env))
    U.append(qU[-1] / (r_scale * s_scale**2))  
    
    # U.append(qU[-1] * r * s * s + random.uniform(-0.1, 0.1))

    ###############################################
    #################### Cotroller ################
    ###############################################
    
    

    ######################### residue disclose #########################

    # 2x(N+2) 크기의 배열 생성
    residue_array = np.zeros((2, env.N + 2), dtype=object)
    # 첫 번째 열에 cresi[-1]을 s**2 곱한 값으로 업데이트 # residue array 는 크기 N+2 리스트 두개
    residue_array[0, 0] = int(round(cresi[-1][0, 0] / (s_scale**2)))  # 첫 번째 행 첫 번째 열
    residue_array[1, 0] = int(round(cresi[-1][1, 0] / (s_scale**2)))  # 두 번째 행 첫 번째 열

    # print(residue_array[0, 0])
    # print(residue_array[1, 0])
    # 여기서 만든 residue array가 Xc 업데이트에 쓰일 예정

    # 2x1 배열만 추가 (첫 번째 열)
    resi.append((1/(r_scale*env.L))*residue_array[:, 0].reshape(2, 1))  # 2x1 배열로 추가
    
    #################################################################
    ######################### state update  #########################
    #################################################################
    
    # plant state update
    Xp.append(A @ Xp[-1] + B @ U[-1])  

    # print("암호화 된 Xc",cX0)
    # print("업데이트 전 Xc: ", r*s*lwe.Dec_res(cX0,sk,env))
    # controller state update
    cX0 = lwe.Mod(F_ @ cX0 + qG @ cY[-1] + qR @ residue_array,env.q) 
    
    # print("residue_array", residue_array) # 이 residue_array는 s*s 만큼 scale downed
    ########### TO DEBUG ################

    # print("iteration:", i+1, "번째")
    # print("Y", Y[-1][0])
    # print("U",U[-1])
    # print("resi",resi[-1])
    # print("residue array",residue_array)

    # print("업데이트 후 Xc: ", r*s*lwe.Dec_res(cX0,sk,env))

    # output masking part update 
    Bx = W @ Bx


    # state difference comparing
    diff_u.append(u_[-1] - U[-1])
    diff_Xc.append(x_c[-1] - Xc[-1])
    
    end_time = time.time()  # 종료 시간 기록
    execution_times.append(end_time - start_time)  # 실행 시간 저장

    ########### TO DEBUG ################

    
avg_execution_time = sum(execution_times) / iter
print(f"\n평균 이터레이션 실행 시간: {avg_execution_time * 1000:.3f} ms")


# Convert lists to arrays for plotting
disturbance_values = np.array(disturbance_values)  # 배열 변환



# Convert lists to arrays for plotting
qU = np.hstack(qU).flatten()
u_ = np.hstack(u_).flatten()
U = np.hstack(U).flatten()
y_ = np.hstack(y_)
Y = np.hstack(Y)
diff_u = np.hstack(diff_u).flatten()
diff_Xc = np.hstack(diff_Xc).flatten()
resi = np.hstack(resi)
time = Ts * np.arange(iter)

# Figure 설정: 3개의 별도 창으로 분리하고 SVG 파일로 저장
# 현재 시간을 파일명에 포함
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Original input (u_) vs Encrypted input (U)
plt.figure(figsize=(8, 4))
plt.plot(time, U, label='$\\tilde{u}(t)$', linestyle='-', color='r')
plt.plot(time, u_, label='$u(t)$', linestyle='--', color='b')
plt.plot(time, disturbance_values, label='$a(t)$', linestyle=':', color='k')
plt.xlabel('Time (sec)')
plt.ylabel('$\\tilde{u}(t)$, $u(t)$, $a(t)$')
plt.legend()
plt.grid(True)
plt.savefig(f'plot1_controller_comparison_{timestamp}.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()

# 2. Difference between u_ and U (절댓값, 어택 전만)
diff_u_plot = np.abs(diff_u)
# 표준정규분포 외란은 계속 있으므로, 추가 외란(2)이 들어가는 구간을 찾아야 함
attack_start_idx = np.argmax(disturbance_values > 0.1)  # 0.1보다 큰 값이 들어가는 첫 인덱스 (추가 외란 구간)
if attack_start_idx == 0:  # 추가 외란이 없으면 전체 플롯
    attack_start_idx = len(time)
plt.figure(figsize=(8, 4))
plt.plot(time[:attack_start_idx], diff_u_plot[:attack_start_idx], label='$|u(t) - \\tilde{u}(t)|$ (before attack)', color='g', linestyle='-')
plt.xlabel('Time (sec)')
plt.ylabel('$|u(t) - \\tilde{u}(t)|$')
plt.legend()
plt.grid(True)
plt.savefig(f'plot2_control_difference_{timestamp}.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()

# 3. Residue Disclosure
plt.figure(figsize=(8, 4))
plt.plot(time, resi[0, :], label='$r_p(t)$', color='m', linestyle='--')
plt.plot(time, resi[1, :], label='$r_\\phi(t)$', color='y', linestyle='-')
plt.xlabel('Time (sec)')
plt.ylabel('$r(t)$')
plt.legend()
plt.grid(True)
plt.savefig(f'plot3_residue_disclosure_{timestamp}.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()

# 4. Plant output y(t)
plt.figure(figsize=(8, 4))
plt.plot(time, Y[0, :], label='$\\tilde{y}_1(t)$', linestyle='-', color='r')
plt.plot(time, y_[0, :], label='$y_1(t)$', linestyle='--', color='b')
plt.plot(time, Y[1, :], label='$\\tilde{y}_2(t)$', linestyle='-', color='m')
plt.plot(time, y_[1, :], label='$y_2(t)$', linestyle='--', color='c')
plt.xlabel('Time (sec)')
plt.ylabel('$y(t)$')
plt.legend()
plt.grid(True)
plt.savefig(f'plot4_plant_output_{timestamp}.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSVG 파일들이 저장되었습니다:")
print(f"- plot1_controller_comparison_{timestamp}.svg")
print(f"- plot2_control_difference_{timestamp}.svg") 
print(f"- plot3_residue_disclosure_{timestamp}.svg")
print(f"- plot4_plant_output_{timestamp}.svg")


# time check
execution_times = np.array(execution_times)
avg_time = np.mean(execution_times)
min_time = np.min(execution_times)
max_time = np.max(execution_times)
std_time = np.std(execution_times)

print(f"\n평균 이터레이션 실행 시간: {avg_time * 1000:.3f} ms")
print(f"최솟값: {min_time * 1000:.3f} ms")
print(f"최댓값: {max_time * 1000:.3f} ms")
print(f"표준편차: {std_time * 1000:.3f} ms")

