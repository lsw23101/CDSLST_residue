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
r_scale = 10000
s_scale = 10000


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
    # xp0 = np.array([[0.05], [0.0], [0.05], [0.0]])
    xp0 = np.array([[-0.1], [-0.1], [0.1], [0.1]])
    # xp0 = np.array([[1], [1], [1], [1]])
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
    # avg_diff_u = np.mean([np.linalg.norm(diff) for diff in diff_u])
    max_diff_u = np.max([np.linalg.norm(diff) for diff in diff_u])
    # # outlier 값 필터링 (0.2 초과하는 값은 제거)
    # if max_diff_u > 0.2:
    #     return None
    return max_diff_u


# print("3593", run_simulation(10000, 3593, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk))
# print("10000", run_simulation(10000, 10000, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk))


# 시뮬레이션 설정
r_fixed_values = [10000]
# 1/s_scale 값을 1/100000부터 1/1000까지 선형적으로 100개 생성
inverse_s_values = np.linspace(1/1000000, 1/1000, num=100)
# s_scale 값은 역수로 계산
s_values = [int(1/inv_s) for inv_s in inverse_s_values]
# 결과 저장
line_results = {r: [] for r in r_fixed_values}
S_points = {r: [] for r in r_fixed_values}
X_points = {r: [] for r in r_fixed_values}

# 시뮬레이션 수행
for r in r_fixed_values:
    for s in s_values:
        print(f"Simulating for r={r}, s={s}...")
        max_diff = run_simulation(r, s, env, lwe, A, B, C, F_, G_, H_, P_, J_, R_, sk)
        # outlier 값이 아닌 경우만 결과에 추가
        if max_diff is not None:
            line_results[r].append(max_diff)
            S_points[r].append(s)
            X_points[r].append(1.0 / s)
            print(f"r_scale={r}, s_scale={s} → avg_diff_u = {max_diff:.6f}")
        else:
            print(f"r_scale={r}, s_scale={s} → outlier (avg_diff_u > 0.2), skipped")

# 시각화
plt.figure(figsize=(8, 6))
for r in r_fixed_values:
    # 해당 r_scale에 대한 결과가 있는 경우만 플롯
    if len(line_results[r]) > 0:
        # x축: 1/s (0~1 구간)
        plt.plot(X_points[r], line_results[r], marker='o', markersize=4, linewidth=2, color='orange', label=f"$\\mathrm{{s_r}} = 1/{r}$")

r_fixed = 10000

# 추가 일차함수: H_inf *Uinv_inf * U_inf * series_inf * (G_inf/2 * r + y_inf * s + 1/2 r s)

H_inf = 150.614
y_inf = 0.28
# y_inf = 1.08
arx_total = 12.5265
residue_inf = 0.1

"""
이하 보조 직선/다항식은 x축을 1/s(=0~1)로 맞춰 계산
"""
# x 축 정의 (시뮬레이션에서 실제 사용된 1/s를 우선 사용, 없으면 inverse_s_values 사용)
x_axis_source = X_points.get(r_fixed, [])
x_axis = sorted(x_axis_source) if len(x_axis_source) > 0 else list(inverse_s_values)

# r은 1/r_fixed 사용
r_eff = 1.0 / r_fixed

# 2차식: H_inf * G_inf * r + (H_inf + G_inf) * r * s + (H_inf + G_inf) * y_inf * s + y_inf * s^2 + r * s^2
# y = const + linear * x + quadratic * x^2 형태
const_term = H_inf * arx_total * r_eff
linear_term = (2* H_inf + arx_total) * (r_eff + 2 * y_inf)
quadratic_term = 8 * y_inf + 4 * r_eff
poly_values = [(const_term + linear_term * x + quadratic_term * x**2) for x in x_axis]

label_text = f"${const_term:.4f} + {linear_term:.4f} \, s_s + {quadratic_term:.4f} \, s_s^2$"
plt.plot(
    x_axis,
    poly_values,
    'k-.',
    linewidth=2,
    label=label_text,
)

# 데이터와 보조 곡선의 차이 평균 (MAE) 계산 및 출력
if X_points[r_fixed] and line_results[r_fixed]:
    x_data = np.array(X_points[r_fixed])
    y_data = np.array(line_results[r_fixed])
    y_fit = const_term + linear_term * x_data + quadratic_term * x_data**2
    mae = float(np.mean(np.abs(y_data - y_fit)))
    print(f"Mean absolute difference (data vs quadratic): {mae:.6e}")

plt.xlabel("$\\mathrm{s_s}$", fontsize=12)
plt.ylabel("$\\sup_{t\\geq0}(u(t) - \\tilde{u}(t))$", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# SVG 파일로 저장
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'figure5_compare_{timestamp}.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSVG 파일이 저장되었습니다: figure4_study_{timestamp}.svg")