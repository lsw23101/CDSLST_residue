import numpy as np
import matplotlib.pyplot as plt
import enc_function_qsize_int64 as lwe
import time
import pandas as pd

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
s = 0.00001
qG = np.round(G_ / s).astype(int)
qH = np.round(H_).astype(int)
qP = np.round(P_ / s).astype(int)
qJ = np.round(J_ / s).astype(int)
qR = np.round(R_ / s).astype(int)
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
    
    start_time = time.time()  # 시작 시간 기록
    
    disturbance = 0 if i != 10 else 0
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
    Y.append(float((C @ Xp[-1]).item()))  # Y에 스칼라 값 저장
    qY.append(int(np.round(Y[-1] / r).astype(int)))
    cY.append(lwe.Enc(qY[-1], sk, env))

    # controller
    cU.append(lwe.Mod(qP @ cX0, env.q))

    # actuator
    qU.append(lwe.Dec(cU[-1], sk, env))
    U.append(qU[-1] * r * s * s)  

    
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

    # controller state update
    cX0 = lwe.Mod(F_ @ cX0 + qG @ cY[-1] + qR @ re_enc_resi[-1],env.q) 

    # state difference comparing
    diff_u.append(u_[-1] - U[-1])
    diff_Xc.append(x_c[-1] - Xc[-1])
    
    end_time = time.time()  # 종료 시간 기록
    execution_times.append(end_time - start_time)  # 실행 시간 저장

    # ########### TO DEBUG ################
    # print("cX0: ", cX0)
    # print("cU: ", cU[-1])
    # print("cY: ", cY[-1])
    # print("re_enc_resi: ", re_enc_resi[-1])
    
avg_execution_time = sum(execution_times) / iter
print(f"\n평균 이터레이션 실행 시간: {avg_execution_time * 1000:.3f} ms")

# 데이터 저장을 위한 DataFrame 생성 (이터레이션 인덱스 사용)
df = pd.DataFrame({'iteration': range(1, iter + 1), 'diff_u': diff_u})

# CSV 파일로 저장
df.to_csv('diff_u_data.csv', index=False)



# Convert lists to arrays for plotting
u_ = np.hstack(u_).flatten()
U = np.hstack(U).flatten()
y_ = np.hstack(y_).flatten()
Y = np.hstack(Y).flatten()
diff_u = np.hstack(diff_u).flatten()
diff_Xc = np.hstack(diff_Xc).flatten()
time = Ts * np.arange(iter)
plt.figure(1)
plt.plot(time, u_, label='original 1')
plt.plot(time, U, label='encrypted 1')
plt.title('Control input and residue')
plt.legend()

plt.figure(2)
plt.plot(time, y_, label='original 1')
plt.plot(time, Y, label='encrypted 1')
plt.title('Plant output y')
plt.legend()

plt.figure(3)
plt.plot(time, diff_u, label='u_ - U')
plt.title('Difference between u_ and U')
plt.legend()


plt.show()

