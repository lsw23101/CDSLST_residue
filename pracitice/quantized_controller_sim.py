# 매트랩과 동일하게 나타남...
# 양자화한 컨트롤러는 peak value는 비슷하지만 ripple은 조금 생김

import numpy as np
import matplotlib.pyplot as plt

#np.seterr(over='raise', invalid='raise')  # 오버플로우 및 NaN 발생 시 에러 발생

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
qG = np.round(G_ / s)
qH = np.round(H_)
qP = np.round(P_ / s)
qJ = np.round(J_ / s)
qR = np.round(R_ / s)
# print("qP:", qP)
# print("qG:", qG)
# print("qR:", qR)
# print("qH:", qH)
# print("qJ:", qJ)

# Simulation settings
iter = 100
xp0 = np.array([[0], [0], [0.005], [0]])
xc0 = np.array([[0], [0], [0], [0]])

# Initialize variables
xp, xc, u, y = [xp0], [xc0], [], []
x_p, x_c, u_, y_, r_ = [xp0], [xc0], [], [], []
Xp, qXc, Xc, Y, U = [xp0], [np.round(xc0 / (r * s))], [xc0], [], []
resi, qY, qU, qresi, residue = [], [], [], [], []
diff_u, diff_Xc = [], []
print("qXc:", qXc[0])  # 처음 값
print("qresi:", qresi)
# Simulation loop
for i in range(iter):
    disturbance = 0 if i != 10 else 0
    print("iteration:", i, "번째")
    # Converted controller
    y_.append(C @ x_p[-1])
    u_.append(P_ @ x_c[-1] + disturbance)
    r_.append(H_ @ x_c[-1] + J_ * y_[-1])
    x_p.append(A @ x_p[-1] + B @ u_[-1])
    x_c.append(F_ @ x_c[-1] + G_ @ y_[-1] + R_ @ r_[-1])

    # Quantized controller
    
    # input output residue ipdate
    Y.append(C @ Xp[-1])
    qY.append(np.round(Y[-1] / r))
    qU.append(qP @ qXc[-1])
    U.append(qU[-1] * r * s * s)
    print("Y", Y[-1])
    
    qresi.append(qH @ qXc[-1] + qJ * qY[-1])
    resi.append(qresi[-1] * s)
    residue.append(r * resi[-1])
    
    # state update
    Xp.append(A @ Xp[-1] + B @ U[-1])
    new_qXc = F_ @ qXc[-1] + qG @ qY[-1] + qR @ resi[-1]
    qXc.append(new_qXc)
    Xc.append(r * s * new_qXc)
    
    # state difference comparing
    diff_u.append(u_[-1] - U[-1])
    diff_Xc.append(x_c[-1] - Xc[-1])

# Convert lists to arrays for plotting
u_ = np.hstack(u_).flatten()
U = np.hstack(U).flatten()
y_ = np.hstack(y_).flatten()
Y = np.hstack(Y).flatten()
diff_u = np.hstack(diff_u).flatten()
diff_Xc = np.hstack(diff_Xc).flatten()
time = Ts * np.arange(iter)

# Plot results
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

# state diff는 출력값이 4개라서 plot하는거 좀 수정...

# plt.figure(4)
# plt.plot(time, diff_Xc)
# plt.title('Difference between x_c and Xc')
# plt.legend(['x_c(1) - Xc(1)', 'x_c(2) - Xc(2)', 'x_c(3) - Xc(3)', 'x_c(4) - Xc(4)'])

plt.show()
