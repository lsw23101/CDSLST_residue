import numpy as np
import matplotlib.pyplot as plt
import enc_function_qsize_int64 as lwe

np.seterr(over='raise', invalid='raise')  # 오버플로우 및 NaN 발생 시 에러 발생

# 환경 객체 생성
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
r = 0.00001
s = 0.00001
qG = np.round(G_ / s).astype(int)
qH = np.round(H_).astype(int)
qP = np.round(P_ / s).astype(int)
qJ = np.round(J_ / s).astype(int)
qR = np.round(R_ / s).astype(int)
# print("qP:", qP)
# print("qG:", qG)
# print("qR:", qR)
# print("qH:", qH)
# print("qJ:", qJ)

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
# # ## ## ## ## ## ## ## ## Simulation settings # ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
iter = 100
xp0 = np.array([[0], [0], [0.005], [0]])
xc0 = np.array([[0], [0], [0], [0]])
# Initialize variables with proper int conversion at the beginning

xp, xc, u, y = [xp0], [xc0], [], []
x_p, x_c, u_, y_, r_ = [xp0], [xc0], [], [], []
Xp, qXc, Xc, Y, U = [xp0], [np.round(xc0 / (r * s)).astype(int)], [xc0], [], []
resi, qY, qU, re_enc_resi, residue, cY, cU, cresi = [], [], [], [], [], [], [], []
diff_u, diff_Xc = [], []

qX0 = np.round(xc0 / (r * s)).astype(int)

# Encrypted controller initial state
cX0 = lwe.EncVec(qX0, sk, env)

# Simulation loop
for i in range(iter):
    disturbance = 0 if i != 10 else 0
    print("iteration:", i+1, "번째")
    ############## Converted controller ####################
    y_.append(C @ x_p[-1])
    u_.append(P_ @ x_c[-1] + disturbance)
    r_.append(H_ @ x_c[-1] + J_ * y_[-1])
    x_p.append(A @ x_p[-1] + B @ u_[-1])
    x_c.append(F_ @ x_c[-1] + G_ @ y_[-1] + R_ @ r_[-1])
    #########################################################
    
    ##### 디버그하기 위해서 cX0를 복호화 #######
    print("X0:", lwe.DecVec(cX0,sk,env))
    
    ############## Encrypted controller ############## 
    
    # sensor
    Y.append(float((C @ Xp[-1]).item()))  # Y에 스칼라 값 저장
    print("y_랑 Y", y_[-1],Y[-1])
    qY.append(int(np.round(Y[-1] / r).astype(int)))
    cY.append(lwe.Enc(qY[-1], sk, env))
    ##### 디버그하기 위해서 cY를 복호화 #######
    # print("cY 복호화한 qY :", lwe.Dec(cY[-1],sk,env))

    # controller
    cU.append(lwe.Mod(qP @ cX0, env.q))
    ##### 디버그하기 위해서 cU를 복호화 #######
    # print("cU 복호화 :", lwe.Dec(cU[-1],sk,env))

    # actuator
    qU.append(lwe.Dec(cU[-1], sk, env))
    
    # 여기서 지금 cU를 계산하고 복호화 했더니 엉뚱한 값이 나오는 상황
    
    U.append(qU[-1] * r * s * s)  # NumPy 대신 Python 기본 리스트 사용

    
    #################################################################
    ######################### re-encryption #########################
    #################################################################
    
    cresi.append(lwe.Mod(qH @ cX0 + qJ * cY[-1], env.q))  # encrypted controller output r
    resi.append(lwe.Dec(cresi[-1],sk,env)) # 복호화 1/sr at actuator
    re_enc_resi.append(lwe.Enc(np.round(float(resi[-1] * s)).astype(int), sk, env))  # 1/s scaled & controller auxiliary input


    #################################################################
    ######################### state update  #########################
    #################################################################
    
    # plant state update
    Xp.append(A @ Xp[-1] + B @ U[-1])  

    # print("before cX0:", cX0)
    # print("before decrypted X0:", lwe.DecVec(cX0, sk, env))
    # 여기까진 오케이
    # controller state update
    ############### 여기서 cX0 구하고 복호화하는게 딱 문제가 생기는 것으로 판단 ####################
    
    # print("cY랑 cR이 둘다 0 암호화 한거", cY[-1],"\n", re_enc_resi[-1])
    # print("행렬들: ", F_, qG, qR)
    
    # print("re_enc_resi[-1] 이거 먼저 복호화 :",lwe.Dec(re_enc_resi[-1], sk, env)) # 범인 이 놈
    
    ## 하나씩 해보자;;
    # print("F_ @ cX0 에 복호화 :",lwe.DecVec(F_ @ cX0, sk, env))
    # print("qG @ cY[-1] 에 복호화 :",lwe.Dec(qG @ cY[-1], sk, env))
    # print("qR @ re_enc_resi[-1] 에 복호화 :",lwe.Dec(qR @ re_enc_resi[-1], sk, env))
    
    cX0 = lwe.Mod(F_ @ cX0 + qG @ cY[-1] + qR @ re_enc_resi[-1],env.q) 
    ##### 디버그 cX0를 복호화 #######
    # print("encrypted X0:", cX0)
    print("updated X0:", lwe.DecVec(cX0,sk,env))
    print("updated x_c:", x_c[-1])
    
    # state difference comparing
    diff_u.append(u_[-1] - U[-1])
    diff_Xc.append(x_c[-1] - Xc[-1])
    
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
    
# Convert lists to arrays for plotting
u_ = np.hstack(u_).flatten()
U = np.hstack(U).flatten()
y_ = np.hstack(y_).flatten()
Y = np.hstack(Y).flatten()
diff_u = np.hstack(diff_u).flatten()
diff_Xc = np.hstack(diff_Xc).flatten()
time = Ts * np.arange(iter)

# # ## ## ## ## ## ## ## ## ## ## # Plot results # ## ## ## ## ## ## ## ## ## ## ## ## #
plt.figure(1)
plt.plot(time, u_, label='original 1')
plt.plot(time, U, label='quantized 1')
plt.title('Control input and residue')
plt.legend()

plt.figure(2)
plt.plot(time, y_, label='original 1')
plt.plot(time, Y, label='quantized 1')
plt.title('Plant output y')
plt.legend()

plt.figure(3)
plt.plot(time, diff_u, label='u_ - U')
plt.title('Difference between u_ and U')
plt.legend()


plt.show()
























# state diff는 출력값이 4개라서 plot하는거 좀 수정...

# plt.figure(4)
# plt.plot(time, diff_Xc)
# plt.title('Difference between x_c and Xc')
# plt.legend(['x_c(1) - Xc(1)', 'x_c(2) - Xc(2)', 'x_c(3) - Xc(3)', 'x_c(4) - Xc(4)'])
