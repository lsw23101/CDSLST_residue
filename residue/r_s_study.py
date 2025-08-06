import numpy as np
import matplotlib.pyplot as plt
import utils.encryption_res_qsize128 as lwe
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


A = np.array([[1.000000000000000, 0.099091315838924, 0.013632351698217, 0.000450408274543], 
          [0.000000000000000, 0.981778666086312, 0.278888611257509, 0.013632351698217], 
          [0.000000000000000, -0.002318427159561, 1.159794783603433, 0.105273449905749], 
          [0.000000000000000, -0.047430035928148, 3.276421050834629, 1.159794783603433]])

B = np.array([[0.009086841610758], 
          [0.182213339136875], 
          [0.023184271595607], 
          [0.474300359281477]])


R_ = np.array([[-0.158717825831511, 0.168218536215741], 
          [-0.162175411950193, 0.333399971252324], 
          [-0.007573452606575, 0.076446003699856], 
          [-0.019672108903376, 0.187645926159056]])

G_ = np.array([[0.203157831946646, -0.031566146482456], 
          [0.007723874130122, 0.138764067572083], 
          [-0.154623874970329, 0.058527812034872], 
          [-0.133834642969481, 0.047360629819598]])

H_ = np.array([[-12.118455534346356, 3.466860377603570, -0.000000000000002, -0.000000000000002], 
          [-5.405051675309685, -1.567040096671678, -0.000000000000001, -0.000000000000001]])

P_ = np.array([[-167.022116552110845, 57.580185408028491, 243.134418436244800, -386.482053445286169]])




# L*r*s*s ~ 2^48 정도
# 40배 정도 더 늘어난거니까

# # Quantization parameters
# r_scale = 100000
# s_scale = 100000


def run_simulation(r_scale, s_scale, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk):
    from decimal import Decimal
    import numpy as np
    import time

    scaled_value = s_scale**2
    inverse = mod_inverse(scaled_value, env.q)

    qG = np.round(G_ * s_scale).astype(int)
    qH = np.round(H_ * s_scale).astype(int)
    qP = np.round(P_ * s_scale).astype(int)
    qJ = np.round(J_ * s_scale**2).astype(object)
    qR = np.round(R_ * s_scale).astype(int)

    J_inv = (inverse * J_).astype(object)

    M = np.vectorize(lambda x: int(lwe.Mod(x, env.q)))(J_inv @ qH)
    W = np.vectorize(lambda x: int(lwe.Mod(x, env.q)))(F_ - qG @ J_inv @ qH)

    iter = 100
    xp0 = np.array([[0.05], [0.0], [0.05], [0.0]])
    xc0 = np.array([[0], [0], [0], [0]])
    xp, xc, u, y = [xp0], [xc0], [], []
    x_p, x_c, u_, y_, r_ = [xp0], [xc0], [], [], []
    Xp, qXc, Xc, Y, U = [xp0], [np.round(xc0 * r_scale * s_scale).astype(int)], [xc0], [], []
    resi, qY, qU, residue, cY, cU, cresi = [], [], [], [], [], [], []
    diff_u, diff_Xc = [], []

    qX0 = np.round(xc0 * r_scale * s_scale).astype(int)
    cX0, Bx = lwe.Enc_state(qX0, sk, env)

    for i in range(iter):
        disturbance = 2 if 100 < i < 500 else 0

        y_.append(C @ x_p[-1])
        u_.append(P_ @ x_c[-1])
        r_.append(H_ @ x_c[-1] + J_ @ y_[-1])
        x_p.append(A @ x_p[-1] + B @ u_[-1])
        x_c.append(F_ @ x_c[-1] + G_ @ y_[-1] + R_ @ r_[-1])

        Y.append(C @ Xp[-1])
        qY.append(np.vectorize(lambda x: int(round(Decimal(x))), otypes=[object])(Y[-1] * r_scale))
        cY.append(lwe.Enc_res(qY[-1], sk, Bx, M, env))

        cU.append(lwe.Mod(qP @ cX0, env.q))
        cU[-1][0][0] += disturbance * int(env.L * r_scale * s_scale**2)

        cresi.append(lwe.Mod(qH @ cX0 + qJ @ cY[-1], env.q))

        qU.append(lwe.Dec_res(cU[-1], sk, env))
        U.append(qU[-1] / (r_scale * s_scale**2))

        residue_array = np.zeros((2, env.N + 2), dtype=object)
        residue_array[0, 0] = int(round(cresi[-1][0, 0] / (s_scale**2)))
        residue_array[1, 0] = int(round(cresi[-1][1, 0] / (s_scale**2)))
        resi.append((1/(r_scale*env.L))*residue_array[:, 0].reshape(2, 1))

        Xp.append(A @ Xp[-1] + B @ U[-1])
        cX0 = lwe.Mod(F_ @ cX0 + qG @ cY[-1] + qR @ residue_array, env.q)
        Bx = W @ Bx

        diff_u.append(u_[-1] - U[-1])
        diff_Xc.append(x_c[-1] - Xc[-1])

    max_diff_u = max([np.linalg.norm(diff) for diff in diff_u])
    return max_diff_u


# print("3593", run_simulation(10000, 3593, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk))
# print("10000", run_simulation(10000, 10000, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk))


# 시뮬레이션 설정
r_fixed_values = [1000, 2000, 5000, 10000, 100000]
s_values = [int(x) for x in np.logspace(3, 6, num=100)]
# 결과 저장
line_results = {r: [] for r in r_fixed_values}

# 시뮬레이션 수행
for r in r_fixed_values:
    for s in s_values:
        print(f"Simulating for r={r}, s={s}...")
        max_diff = run_simulation(r, s, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk)
        line_results[r].append(max_diff)
        print(f"r_scale={r}, s_scale={s} → max_diff_u = {max_diff:.6f}")

# 시각화
plt.figure()
for r in r_fixed_values:
    plt.plot(s_values, line_results[r], marker='o', markersize=4, linewidth=1.5, label=f"1/r = {1/r}")

plt.axhline(0.01, color='gray', linestyle='-', linewidth=2, label='Threshold = 0.01')

plt.xscale('log')
plt.xlabel("1/s (log scale)")
plt.ylabel("$\\sup_t(u_{original}(t) - u_{encrypted}(t))$")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()