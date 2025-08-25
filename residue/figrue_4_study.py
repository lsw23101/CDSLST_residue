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
r_scale = 100000
s_scale = 100000


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
    xp0 = np.array([[-0.1], [-0.1], [0.1], [0.1]])
    xc0 = np.array([[0.0], [0.0], [0.0], [0.0]])
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

    # u diff 값들의 평균 계산
    avg_diff_u = np.mean([np.linalg.norm(diff) for diff in diff_u])
    # outlier 값 필터링 (0.2 초과하는 값은 제거)
    if avg_diff_u > 0.2:
        return None
    return avg_diff_u


# print("3593", run_simulation(10000, 3593, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk))
# print("10000", run_simulation(10000, 10000, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk))


# 시뮬레이션 설정
r_fixed_values = [1000, 2000, 5000, 10000, 100000]
# 1/s_scale 값을 1/100000부터 1/1000까지 선형적으로 100개 생성
inverse_s_values = np.linspace(1/100000, 1/1000, num=100)
# s_scale 값은 역수로 계산
s_values = [int(1/inv_s) for inv_s in inverse_s_values]
# 결과 저장
line_results = {r: [] for r in r_fixed_values}

# 시뮬레이션 수행
for r in r_fixed_values:
    for s in s_values:
        print(f"Simulating for r={r}, s={s}...")
        max_diff = run_simulation(r, s, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk)
        # outlier 값이 아닌 경우만 결과에 추가
        if max_diff is not None:
            line_results[r].append(max_diff)
            print(f"r_scale={r}, s_scale={s} → avg_diff_u = {max_diff:.6f}")
        else:
            print(f"r_scale={r}, s_scale={s} → outlier (avg_diff_u > 0.2), skipped")

# 시각화
plt.figure(figsize=(12, 8))
for r in r_fixed_values:
    # 해당 r_scale에 대한 결과가 있는 경우만 플롯
    if len(line_results[r]) > 0:
        # s_scale 값에 대응하는 1/s_scale 값 계산
        r_s_values = [s for s in s_values if run_simulation(r, s, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk) is not None]
        r_inverse_s_values = [1/s for s in r_s_values]
        plt.plot(r_inverse_s_values, line_results[r], marker='o', markersize=6, linewidth=2, label=f"$\\mathrm{{s_r}} = 1/{r}$")

plt.xlabel("$\\mathrm{s_s}$", fontsize=14)
plt.ylabel("$\\frac{1}{T} \\int_0^T |u(t) - \\tilde{u}(t)| dt$", fontsize=14)
plt.title("Impact of $\\mathrm{s_s}$ on Average Difference for Fixed $1/\\mathrm{s_r}$ Values", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# SVG 파일로 저장
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'figure4_study_{timestamp}.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSVG 파일이 저장되었습니다: figure4_study_{timestamp}.svg")