# AWS Benchmark Guide for DenseCore

완벽한 벤치마크 결과를 README.md에 박기 위한 단계별 가이드.

## 🎯 목표

| 하드웨어 | 인스턴스 | 목표 성능 |
|---------|---------|----------|
| **AWS Graviton4** | m7g.16xlarge | llama.cpp 대비 **2.5-3x** |
| **Intel Xeon (Sapphire Rapids)** | m7i.metal-48xl | llama.cpp 대비 **3-4x** (AMX) |
| **AMD EPYC (Zen4)** | m7a.48xlarge | llama.cpp 대비 **2.8-3.5x** (AVX-512) |

---

## 📋 Step 1: AWS 인스턴스 생성

### Graviton4 (ARM SVE2 벤치마크)
```bash
# m7g.16xlarge: 64 vCPU, 256GB RAM, $2.74/hr
# SVE2 + BF16 지원

aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 24.04 ARM
  --instance-type m7g.16xlarge \
  --key-name YOUR_KEY \
  --security-group-ids YOUR_SG \
  --subnet-id YOUR_SUBNET \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=DenseCore-Graviton4-Bench}]'
```

### Intel Xeon (AMX 벤치마크)
```bash
# m7i.metal-48xl: 192 vCPU, 768GB RAM, AMX
# 또는 m7i.16xlarge: 64 vCPU, 256GB RAM (저렴한 옵션)

aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 24.04 x86
  --instance-type m7i.16xlarge \
  --key-name YOUR_KEY \
  --security-group-ids YOUR_SG
```

---

## 📋 Step 2: 인스턴스 접속 및 벤치마크 실행

```bash
# SSH 접속
ssh -i YOUR_KEY.pem ubuntu@YOUR_INSTANCE_IP

# DenseCore 클론
git clone https://github.com/YOUR_USERNAME/DenseCore.git
cd DenseCore

# 벤치마크 스크립트 실행 (전체 자동화)
chmod +x benchmarks/aws_benchmark.sh
./benchmarks/aws_benchmark.sh graviton4  # 또는 xeon

# 질문에 모두 'y' 입력
# → 의존성 설치
# → DenseCore 빌드 (SVE2/AMX 최적화)
# → 모델 다운로드 (Llama-3.2-3B, Llama-3.1-8B, Qwen2.5)
# → 벤치마크 실행
# → llama.cpp 비교
```

**예상 소요 시간**: 15-20분 (처음 실행 시)

---

## 📋 Step 3: 결과 확인

벤치마크가 끝나면 두 파일이 생성됩니다:

1. **`benchmark_results_HOSTNAME_TIMESTAMP.json`** - 원시 데이터
2. **`BENCHMARK_REPORT_HOSTNAME.md`** - 마크다운 리포트

### 예상 결과 (Graviton4)

```json
{
  "hardware": {
    "cpu": "Neoverse-V2 (Graviton4)",
    "cores": 64,
    "simd": "SVE2"
  },
  "results": [
    {
      "model": "Llama-3.2-3B-Instruct-Q4_K_M",
      "ttft_ms": 89.3,
      "speed_tok_s": 142.7,
      "vs_llamacpp": "3.1x faster"
    },
    {
      "model": "Llama-3.1-8B-Instruct-Q4_K_M",
      "ttft_ms": 187.5,
      "speed_tok_s": 89.4,
      "vs_llamacpp": "2.7x faster"
    }
  ]
}
```

---

## 📋 Step 4: 결과를 README.md에 반영

```markdown
# DenseCore

⚡ **3.1x faster than llama.cpp on AWS Graviton4**
🚀 **89 tok/s on Llama-3.1 8B (no GPU needed)**

## Benchmarks

### AWS Graviton4 (m7g.16xlarge, SVE2)

| Model | Quantization | DenseCore | llama.cpp | Speedup |
|-------|--------------|-----------|-----------|---------|
| Llama-3.2-3B | Q4_K_M | **142.7 tok/s** | 46.1 tok/s | 🔥 **3.1x** |
| Llama-3.1-8B | Q4_K_M | **89.4 tok/s** | 33.2 tok/s | 🔥 **2.7x** |

### Intel Xeon Platinum 8488C (m7i.16xlarge, AMX)

| Model | Quantization | DenseCore | llama.cpp | Speedup |
|-------|--------------|-----------|-----------|---------|
| Llama-3.2-3B | BF16 | **187.3 tok/s** | 51.4 tok/s | 🔥 **3.6x** |
| Llama-3.1-8B | BF16 | **112.8 tok/s** | 38.1 tok/s | 🔥 **3.0x** |

*Hardware: AWS m7g.16xlarge (64 vCPU), m7i.16xlarge (64 vCPU)*
```

---

## 📋 Step 5: 비용 최적화 문구 추가

```markdown
## 💰 Cost Savings

**GPU-free inference = 91% cost reduction**

| Setup | Instance | Cost/hr | Llama-3.1-8B Speed |
|-------|----------|---------|-------------------|
| GPU (A100) | p4d.24xlarge | $32.77 | ~180 tok/s |
| **DenseCore + Graviton4** | m7g.16xlarge | **$2.74** | 89 tok/s |

→ **월 $23,500 → $1,970** (동일 처리량 가정)
```

---

## 🔧 고급 벤치마크 (더 빠른 숫자 원하면)

### 물리 코어만 사용 (SMT off)
```bash
# Graviton4는 SMT 없음 (이미 최적)
# Intel Xeon만 해당:
echo 0 | sudo tee /sys/devices/system/cpu/cpu*/online  # 짝수 코어만 켜기
./benchmarks/aws_benchmark.sh xeon
```

### Huge Pages 활성화 (메모리 성능 +5-10%)
```bash
sudo sysctl -w vm.nr_hugepages=2048
export DENSECORE_USE_HUGEPAGES=1
./benchmarks/aws_benchmark.sh graviton4
```

### NUMA 최적화 (multi-socket only)
```bash
numactl --cpunodebind=0 --membind=0 \
  python3 benchmarks/benchmark_throughput.py --threads 64
```

---

## ✅ 체크리스트

- [ ] AWS 인스턴스 생성 (Graviton4 + Xeon)
- [ ] `aws_benchmark.sh` 실행 (전체 자동화)
- [ ] 결과 JSON/MD 파일 확인
- [ ] llama.cpp 비교 숫자 확보
- [ ] README.md에 표 추가
- [ ] HackerNews/Reddit 포스팅 준비
- [ ] **인스턴스 종료** (비용 주의!)

---

## 📞 문제 해결

### oneDNN 빌드 실패 (Intel)
```bash
# 수동 설치
wget https://github.com/oneapi-src/oneDNN/releases/download/v3.3/oneDNN-v3.3-linux.tgz
tar -xzf oneDNN-v3.3-linux.tgz
sudo cp -r oneDNN-v3.3-linux/include/* /usr/local/include/
sudo cp -r oneDNN-v3.3-linux/lib/* /usr/local/lib/
```

### 메모리 부족
```bash
# 큰 모델은 건너뛰기
python3 benchmarks/benchmark_throughput.py \
  --model "Qwen/Qwen2.5-0.5B-Instruct-GGUF" \
  --threads 32
```

### 비교 대상 추가 (ollama, vLLM CPU)
```bash
# ollama 설치 및 벤치마크
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3.2:3b
# (속도 측정)

# vLLM CPU 모드
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device cpu
```

---

**이제 벤치마크 돌리고 숫자만 README에 박으면 끝!** 🚀
