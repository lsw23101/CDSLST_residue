# CDSLST_residue

이 워크스페이스는 제어 시스템, 암호화, 잔차(residue) 기반 시뮬레이션 및 공격 실험을 위한 코드와 실험 환경을 포함합니다.

- 제어 시스템(예: MIMO Cartpole 등) 시뮬레이션
- LWE 기반 암호화 및 복호화 실험
- residue(잔차) 기반 공격/방어 실험
- lattice-estimator를 통한 LWE 파라미터 보안레벨 분석

---

## 최근 lattice-estimator 보안레벨 결과 (2024-06)

- 예시 파라미터: N=3000, q=18446744073709551557, r=10
- 주요 공격별 rop(보안 비트):
    - usvp: rop ≈ 2^162.2
    - bdd: rop ≈ 2^161.2
    - dual: rop ≈ 2^164.4
    - dual_hybrid: rop ≈ 2^161.1
    - bkw: rop ≈ 2^534.3 (현실적 공격 아님)
    - arora-gb: rop ≈ 2^354.4 (현실적 공격 아님)
- **실질적 보안 비트: 약 161~164비트 (매우 안전)**
- 평균 이터레이션 실행 시간: 10.147 ms

---

# To Do:

- 현재 siso로 구현했을때에는 quantization 파라미터가 너무 작게 나옴
- 그에 따라 q size도 크게 나옴
- multi output system 상황으로 확장시켜서...
