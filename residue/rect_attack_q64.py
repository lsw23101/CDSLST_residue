import numpy as np
import matplotlib.pyplot as plt
import utils.encryption_res_qsize64 as lwe
import time
import random
from decimal import Decimal
from sympy import mod_inverse
from sympy import isprime

np.seterr(over='raise', invalid='raise')  # 오버플로우 및 NaN 발생 시 에러 발생

# 파라미터 생성 // q L e 
Ts = 0.05 # 루프타임이 28ms 니까 50ms 샘플링타임으로 설정
env = lwe.params()  # 환경 설정
sk = lwe.Seret_key(env)
print(isprime(env.q))

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


R_ = np.array([[-0.292818598882407, 0.164982922756992], 
          [-0.249422018446482, 0.227680322106045], 
          [-0.005943194689578, 0.041644608179790], 
          [-0.017141423196389, 0.104455055130682]])

G_ = np.array([[0.402608485515628, -0.023377805376137], 
          [0.006485328448349, 0.349991124035727], 
          [-0.290056249390091, 0.100201763588302], 
          [-0.223997396898498, 0.051258437178160]])

H_ = np.array([[-6.109392087854962, 1.499285882716716, 0.000000000000000, 0.000000000000000], 
          [-2.787670773969834, -1.283777072706201, 0.000000000000000, 0.000000000000000]])

P_ = np.array([[-89.849233365241602, -23.582355805480020, 498.930855552969945, -663.222704330903525]])



# L*r*s*s ~ 2^48 정도
# 40배 정도 더 늘어난거니까

# Quantization parameters
r_scale = 10000
s_scale = 13000

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

xp0 = np.array([[0], [0], [0.1], [0.1]])
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


for i in range(iter):
    
    start_time = time.time()  # 시작 시간 기록 
    
    # 외부 impulse 어택을 400 이터레이션 때
    disturbance = 0
    if i > 200 and i <500:
        disturbance = 2

    disturbance_values.append(disturbance)  # disturbance 저장
    
    '''############# original 컨트롤러 ############## '''

    y_.append(C @ x_p[-1])
    u_.append(P_ @ x_c[-1])
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
    cU[-1][0][0] += disturbance * int(env.L * r_scale * s_scale**2)
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

# Figure 설정: 가로로 길게 (1열 3행)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))
# 예: time이 0~10까지 1000포인트라면 xticks를 간격 2로 설정
xticks = np.arange(0, time[-1] + 1, 2)

# 1. Original input (u_) vs Encrypted input (U)
axes[0].plot(time, U, label='Encrypted Controller', linestyle='-', color='r')
axes[0].plot(time, u_, label='Original Controller', linestyle='--', color='b')
axes[0].plot(time, disturbance_values, label='Attack', linestyle=':', color='k')
axes[0].set_title('Control Input Comparison', fontsize=14)
axes[0].set_yticks([2, 0, -2, -4, -6])
axes[0].tick_params(axis='y', labelsize=12)
axes[0].legend()

# 2. Difference between u_ and U (절댓값, 클립)
diff_u_plot = np.abs(diff_u)  # 절댓값으로 변경
# diff_u_plot = np.clip(diff_u_plot, 0, 0.03)  # 0.2로 클립

# 전체 실선 그리기 (400 이후는 어차피 0이라 위 점선보다 늦게 그리면 가려버림)
# axes[1].plot(time, diff_u_plot, label='||u_diff||', color='g', linestyle='-')

# axes[1].set_title('Norm of Difference', fontsize=14)
# # axes[1].set_yticks([0, 0.01, 0.01])  # y축 범위를 0, 0.1, 0.2로 설정
# axes[1].tick_params(axis='y', labelsize=12)
# axes[1].legend()

# 3. Residue Disclosure
axes[1].plot(time, resi[0, :], label='Residue of y_1', color='m', linestyle='--')
axes[1].plot(time, resi[1, :], label='Residue of y_2', color='y', linestyle='-')
axes[1].set_title('Residue Disclosure', fontsize=14)
axes[1].set_yticks([0.3, 0.2, 0.1, 0, -0.1])
axes[1].tick_params(axis='y', labelsize=12)
axes[1].legend()

# # 별도 Figure에 qU만 그리기
# plt.figure(figsize=(6, 3))
# plt.plot(time, qU, label='Quantized Encrypted Input', color='purple')
# plt.title('Quantized Control Input (qU)', fontsize=14)
# plt.xlabel('Time (s)')
# plt.ylabel('qU')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# Layout 조정 및 플롯 표시
plt.tight_layout(rect=[0, 0, 1, 0.97])  # 위쪽 여백 조정
plt.show()