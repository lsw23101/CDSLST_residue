import numpy as np
import pickle
from decimal import Decimal

# q64 환경 설정
class params_q64:
    def __init__(self):
        self.p = int(2**54)  # p 
        self.L = int(2**10)  # L 
        self.r = 10         # 오류 범위
        self.N = 4000     # 키 차원 
        self.q = self.p * self.L - 59 # 2^64 근처 소수 18446744073709551557

# q128 환경 설정
class params_q128:
    def __init__(self):
        self.p = int(2**118)  # p 
        self.L = int(2**10)  # L 
        self.r = 10         # 오류 범위
        self.N = 8000    # 키 차원 
        self.q = self.p * self.L - 159 # 근처 소수

def Mod(x, p):
    y = np.mod(x, p)  # 요소별 모듈러 계산
    y -= (y >= p / 2) * p  # 범위 조정 (biased modular)
    return y

def Seret_key(env):
    import random
    return np.array([[random.choice([-1, 0, 1])] for _ in range(env.N)], dtype=object)

# 암호화 함수 (rect_attack_q64.py와 동일)
def Enc_res(m, sk, Bx, M, env):
    import random
    n = 2  # 메시지는 y 가 2차원

    A = np.array([[random.randint(0, env.q - 1) for _ in range(env.N)] for _ in range(n)], dtype=object)
    A = Mod(A, env.q)

    e = np.random.normal(0, env.r, size=(n, 1)).astype(int)
    
    mask = M @ Bx
    k = Mod(A @ sk + e + mask, env.q)

    ciphertext = Mod(np.hstack((-mask + env.L*m , A, k)), env.q)
    return ciphertext

def test_q64():
    print("=== q64 환경 테스트 ===")
    
    # 1. 환경 설정
    env = params_q64()
    sk = Seret_key(env)
    
    # 2. 고정된 y 데이터 생성
    Y_data = np.array([[0.05], [0.03]])  # 고정된 y 값
    print("1. 원본 y 데이터:")
    print(f"   Y_data = {Y_data}")
    print(f"   형태: {Y_data.shape}")
    print(f"   타입: {Y_data.dtype}")
    
    # 3. 스케일링 및 정수 변환
    r_scale = 10000
    s_scale = 10000
    qY_data = np.vectorize(lambda x: int(round(Decimal(x))), otypes=[object])(Y_data * r_scale)
    print(f"\n2. 스케일링된 y 데이터 (r_scale={r_scale}):")
    print(f"   qY_data = {qY_data}")
    print(f"   형태: {qY_data.shape}")
    print(f"   타입: {qY_data.dtype}")
    
    # 4. 마스킹 파라미터 설정 (출력 생략)
    M = np.array([[1000, 2000], [3000, 4000]], dtype=object)
    Bx = np.array([[5000], [6000]], dtype=object)
    
    # 5. 암호화
    cY_data = Enc_res(qY_data, sk, Bx, M, env)
    print(f"\n4. 암호화된 y 데이터:")
    print(f"   cY_data 형태: {cY_data.shape}")
    print(f"   cY_data 타입: {cY_data.dtype}")
    print(f"   cY_data 크기: {cY_data.size} 개 요소")
    print(f"   첫 번째 행: {cY_data[0, :5]}...")  # 처음 5개만 출력
    print(f"   두 번째 행: {cY_data[1, :5]}...")  # 처음 5개만 출력
    
    # 6. Pickle로 변환
    pickle_bytes = pickle.dumps(cY_data)
    print(f"\n5. Pickle 변환 결과:")
    print(f"   바이트 크기: {len(pickle_bytes):,} bytes ({len(pickle_bytes)/1024:.1f} KB)")
    print(f"   첫 50바이트 (hex): {pickle_bytes[:50].hex()}")
    print(f"   마지막 50바이트 (hex): {pickle_bytes[-50:].hex()}")
    
    # 7. 역변환 테스트
    restored_data = pickle.loads(pickle_bytes)
    is_identical = np.array_equal(cY_data, restored_data)
    print(f"\n6. 역변환 테스트:")
    print(f"   데이터 무결성: {'✅ 성공' if is_identical else '❌ 실패'}")
    
    # 8. 전송 시간 예측
    size_bytes = len(pickle_bytes)
    print(f"\n7. 전송 시간 예측:")
    print(f"   100Mbps: {size_bytes*8/100e6*1000:.1f} ms")
    print(f"   1Gbps: {size_bytes*8/1e9*1000:.1f} ms")
    print(f"   10Gbps: {size_bytes*8/10e9*1000:.1f} ms")
    
    return size_bytes

def test_q128():
    print("\n=== q128 환경 테스트 ===")
    
    # 1. 환경 설정
    env = params_q128()
    sk = Seret_key(env)
    
    # 2. 고정된 y 데이터 생성
    Y_data = np.array([[0.05], [0.03]])  # 고정된 y 값
    print("1. 원본 y 데이터:")
    print(f"   Y_data = {Y_data}")
    print(f"   형태: {Y_data.shape}")
    print(f"   타입: {Y_data.dtype}")
    
    # 3. 스케일링 및 정수 변환
    r_scale = 10000
    s_scale = 10000
    qY_data = np.vectorize(lambda x: int(round(Decimal(x))), otypes=[object])(Y_data * r_scale)
    print(f"\n2. 스케일링된 y 데이터 (r_scale={r_scale}):")
    print(f"   qY_data = {qY_data}")
    print(f"   형태: {qY_data.shape}")
    print(f"   타입: {qY_data.dtype}")
    
    # 4. 마스킹 파라미터 설정 (출력 생략)
    M = np.array([[1000, 2000], [3000, 4000]], dtype=object)
    Bx = np.array([[5000], [6000]], dtype=object)
    
    # 5. 암호화
    cY_data = Enc_res(qY_data, sk, Bx, M, env)
    print(f"\n4. 암호화된 y 데이터:")
    print(f"   cY_data 형태: {cY_data.shape}")
    print(f"   cY_data 타입: {cY_data.dtype}")
    print(f"   cY_data 크기: {cY_data.size} 개 요소")
    print(f"   첫 번째 행: {cY_data[0, :5]}...")  # 처음 5개만 출력
    print(f"   두 번째 행: {cY_data[1, :5]}...")  # 처음 5개만 출력
    
    # 6. Pickle로 변환
    pickle_bytes = pickle.dumps(cY_data)
    print(f"\n5. Pickle 변환 결과:")
    print(f"   바이트 크기: {len(pickle_bytes):,} bytes ({len(pickle_bytes)/1024:.1f} KB)")
    print(f"   첫 50바이트 (hex): {pickle_bytes[:50].hex()}")
    print(f"   마지막 50바이트 (hex): {pickle_bytes[-50:].hex()}")
    
    # 7. 역변환 테스트
    restored_data = pickle.loads(pickle_bytes)
    is_identical = np.array_equal(cY_data, restored_data)
    print(f"\n6. 역변환 테스트:")
    print(f"   데이터 무결성: {'✅ 성공' if is_identical else '❌ 실패'}")
    
    # 8. 전송 시간 예측
    size_bytes = len(pickle_bytes)
    print(f"\n7. 전송 시간 예측:")
    print(f"   100Mbps: {size_bytes*8/100e6*1000:.1f} ms")
    print(f"   1Gbps: {size_bytes*8/1e9*1000:.1f} ms")
    print(f"   10Gbps: {size_bytes*8/10e9*1000:.1f} ms")
    
    return size_bytes

def main():
    print("=== q64 vs q128 Pickle 변환 크기 비교 ===\n")
    
    # q64 테스트
    size_q64 = test_q64()
    
    # q128 테스트
    size_q128 = test_q128()
    
    # 비교 결과
    print(f"\n=== 비교 결과 ===")
    print(f"q64 크기: {size_q64:,} bytes ({size_q64/1024:.1f} KB)")
    print(f"q128 크기: {size_q128:,} bytes ({size_q128/1024:.1f} KB)")
    
    if size_q128 > size_q64:
        size_diff = size_q128 - size_q64
        size_diff_percent = (size_diff / size_q64) * 100
        print(f"q128이 q64보다 {size_diff:,} bytes ({size_diff_percent:.1f}%) 큽니다")
    else:
        size_diff = size_q64 - size_q128
        size_diff_percent = (size_diff / size_q128) * 100
        print(f"q64가 q128보다 {size_diff:,} bytes ({size_diff_percent:.1f}%) 큽니다")
    
    print(f"\n=== 분석 완료 ===")

if __name__ == "__main__":
    main()
