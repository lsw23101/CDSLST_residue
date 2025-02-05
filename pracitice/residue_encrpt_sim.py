import numpy as np
import matplotlib.pyplot as plt
import residue_encryption as lwe
import time

np.seterr(over='raise', invalid='raise')  # 오버플로우 및 NaN 발생 시 에러 발생

# 파라미터 생성 // q L e 

env = lwe.params()  # 환경 설정
sk = lwe.Seret_key(env)

# Sampling time
Ts = 0.01

############ Plant Model #################
A = np.array([[0,1.0000,0,0],
              [0, -0.18181818, 2.6727272727, 0],
              [0, 0, 0, 1],
              [0, -0.454545454545455, 31.181818181818183 , 0]])  

B = np.array([[0], [1.818181818181818], [0], [4.545454545454545]])  

C = np.array([[1, 0, 0, 0]])

############ Controller Model with Re-Enc #################
P_ = np.array([[0.000170111406236, -0.700242543381168, -0.080579410802687, -0.024922811916774]])  

F_ = np.array([[0, 0, 0, 0],
               [1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]])  

G_ = np.array([[0], [-0.000891204184009], [-0.038783175990242], [0.027732472262777]])  

R_ = np.array([[3557.595247666494], [9.375013335514], [-31.124614622107], [0.0605897659410]])  

H_ = np.array([[0, 0, 0, 1]])  

J_ = 1  


# Quantization parameters
r = 0.000001
s = 0.000001
qG = np.round(G_ / s).astype(int)
qH = np.round(H_).astype(int)
qP = np.round(P_ / s).astype(int)
qJ = np.round(J_ / s).astype(int)
qR = np.round(R_ / s).astype(int)


############ Equivalent input with XC = [0 0 0.001 0]  ##########
J_inv = 4352491850465363121987 # GPT로 구한 inverse....


M = J_inv * qH  # 원본 계산
W = F_ - qG @ (J_inv * qH)  # 원본 계산



print("M", M)
print("W", W)


print("To check multiplicative inverse of J: ", bool(lwe.Mod(J_inv*qJ, env.q)))


print("qP:", qP)
print("qG:", qG)
print("qR:", qR)
print("qH:", qH)
print("qJ:", qJ)


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
# # ## ## ## ## ## ## ## ## Simulation settings # ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #

iter = 100
execution_times = []  # 실행 시간을 저장할 리스트

xp0 = np.array([[0], [0], [0.005], [0]])
xc0 = np.array([[0], [0], [0.001], [0]])
# Initialize variables with proper int conversion at the beginning

xp, xc, u, y = [xp0], [xc0], [], []
x_p, x_c, u_, y_, r_ = [xp0], [xc0], [], [], []
Xp, qXc, Xc, Y, U = [xp0], [np.round(xc0 / (r * s)).astype(int)], [xc0], [], []
resi, qY, qU, residue, cY, cU, cresi = [], [], [], [], [], [], []
diff_u, diff_Xc = [], []

qX0 = np.round(xc0 / (r * s)).astype(int)

# Encrypted controller initial state
cX0, Bx = lwe.EncVec_res(qX0, sk, env)  # 여긱서 암호화 할때 cX0 의 마스킹 파트도 뽑아냄
# Bx  << 이 친구도 이제 state 처럼 sensor 가 가지고 업데이트 시켜줘야 할 요소

########### 여기서 초기값 암호화 한 Bx 들고 와서 MW(t) 곱해서 y를 암호화 ###############
# B'y = By + M W^t Bx
##

# Simulation loop

for i in range(iter):
    
    start_time = time.time()  # 시작 시간 기록 
    
    # 50번째 이터레이션에서 disturbance 값을 1로 설정
    disturbance = 0
    if i == 10:
        disturbance = 0
    elif i == 49:  # 50번째 이터레이션에 disturbance 값을 1로 설정
        disturbance = 0.001
    print("iteration:", i+1, "번째")
    
    #########################################################
    ############## Converted controller ####################
    y_.append(C @ x_p[-1])
    u_.append(P_ @ x_c[-1] + disturbance)
    r_.append(H_ @ x_c[-1] + J_ * y_[-1])
    x_p.append(A @ x_p[-1] + B @ u_[-1])
    x_c.append(F_ @ x_c[-1] + G_ @ y_[-1] + R_ @ r_[-1])
    #########################################################
    #########################################################
    


    #########################################################
    ############## Encrypted controller ############## 
    
    # sensor
    ########## 센서가 할일 W 곱해주는건 마지막에 state update 에서 
    # 암호화 cY.append(lwe.Enc_res(qY[-1], Bx, sk, env))
    # cY [qY+M W Bx , Ay , By + M W Bx]
    # 이런식으루~

    Y.append(float((C @ Xp[-1]).item()))  # Y에 스칼라 값 저장
    qY.append(int(np.round(Y[-1] / r).astype(int)))
    cY.append(lwe.Enc_res(qY[-1], sk, Bx, M,env))

    # controller
    cU.append(lwe.Mod(qP @ cX0, env.q))
    cresi.append(lwe.Mod(qH @ cX0 + qJ * cY[-1], env.q))  # encrypted controller output r
    # 여기서 나오는 cresi는 이론상 1/sr 로 스케일링 된 마스킹 파트가 없는 값
    # cresi [1/sr resi , A, B]

    # actuator
    qU.append(lwe.Dec_res(cU[-1], sk, env))
    U.append(qU[-1] * r * s * s + disturbance)  

    

    ######################### residue disclose #########################

    # 1x(N+2) 크기의 배열 생성
    residue_array = np.zeros((1, env.N + 2), dtype=object)
    residue_array[0, 0] = np.round(float(cresi[-1][0, 0] * s)).astype(int)  # 첫 번째 요소 사용


    # 리스트에 추가
    resi.append(residue_array)

    #################################################################
    ######################### state update  #########################
    #################################################################
    
    # plant state update
    Xp.append(A @ Xp[-1] + B @ U[-1])  

    # controller state update
    cX0 = lwe.Mod(F_ @ cX0 + qG @ cY[-1] + qR @ resi[-1],env.q) 

    # output masking part update 
    Bx = W @ Bx


    # state difference comparing
    diff_u.append(u_[-1] - U[-1])
    diff_Xc.append(x_c[-1] - Xc[-1])
    
    end_time = time.time()  # 종료 시간 기록
    execution_times.append(end_time - start_time)  # 실행 시간 저장

    ########### TO DEBUG ################
    print("cX0: ", cX0)
    print("cU: ", cU[-1])
    print("cY: ", cY[-1])
    print("resi: ", resi[-1])
    
avg_execution_time = sum(execution_times) / iter
print(f"\n평균 이터레이션 실행 시간: {avg_execution_time * 1000:.3f} ms")



# Convert lists to arrays for plotting
u_ = np.hstack(u_).flatten()
U = np.hstack(U).flatten()
y_ = np.hstack(y_).flatten()
Y = np.hstack(Y).flatten()
diff_u = np.hstack(diff_u).flatten()
diff_Xc = np.hstack(diff_Xc).flatten()
# Convert resi to a 1D numpy array containing only the first element of each 1x6 array
resi_flat = np.array([i[0, 0]*r for i in resi])  # 각 1x6 배열에서 첫 번째 값만 추출


time = Ts * np.arange(iter)

# Plotting
plt.figure(1)
plt.plot(time, U, label='encrypted 1')
plt.title('encrpted input')
plt.legend()

plt.figure(2)
plt.plot(time, y_, label='original 1')
plt.title('original output y')
plt.legend()

plt.figure(3)
plt.plot(time, u_, label='original 1')
plt.title('original input u')
plt.legend()

plt.figure(4)
plt.plot(time, resi_flat, label='residue')  # residue 플로팅
plt.title('Residue Disclosure')
plt.legend()

plt.figure(5)
plt.plot(time, diff_u, label='u_ - U')
plt.title('Difference between u_ and U')
plt.legend()

plt.figure(6)
plt.plot(time, Y, label='encrypted 1')
plt.title('encrypted output Y')
plt.legend()

plt.show()