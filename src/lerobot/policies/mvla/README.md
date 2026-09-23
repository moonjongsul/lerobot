# MVLA

SmolVLA + 상태 인식기 + 태스크 플래너 + value 기반 실패 탐지.

```
관측 ──→ [인식기] ──→ [플래너] ──→ 프롬프트 ──→ [MVLA 정책] ──→ 행동
          0.3Hz        규칙          ↑            30Hz
          독립 모델                  └──── [매니저] ←── value 헤드 (30Hz)
                                            경계 검증 · 실패 시 리프롬프트
```

## 구성

| 모듈 | 역할 |
|---|---|
| `data/segments.py` | 서브태스크 런 추출. 모든 라벨의 출처 |
| `data/derive.py` | 중립 프롬프트, value 타깃, 경과시간 |
| `recognizer/` | 동결 인코더 + 선형 프로브 (`mat_state`, `tray_placement`) |
| `planner/rules.py` | 상태 → 프롬프트 + 서브태스크 시퀀스 |
| `planner/manager.py` | 경계 검증, 리프롬프트, 재시도 상한 |
| `configuration_mvla.py` | `MVLAConfig` (SmolVLAConfig 확장) |
| `heads.py` | subtask / value×2 / status 헤드 |
| `processor_mvla.py` | 프롬프트 조립 + dropout |
| `modeling_mvla.py` | `MVLAPolicy` |

---

## 1. 인식기 학습

```bash
python -m lerobot.policies.mvla.recognizer.train \
  --dataset /workspace/m.ax/datasets/lerobot/xarm7_kitting_260923 \
  --out /workspace/m.ax/outputs/mvla_recognizer
```

| 인자 | 기본값 |
|---|---|
| `--dataset` | (필수) LeRobot 데이터셋 |
| `--out` | (필수) 출력 |
| `--cache` | `<out>/frames` — 디코딩 프레임 캐시 |
| `--backbone` | `facebook/dinov2-base` |
| `--image-size` | 224 |
| `--blocks` | 5 (연속 블록 CV) |
| `--skip-placebo` | 대조 생략 (**권장하지 않음**) |

라벨은 **에피소드 구조에서 자동 추출**되므로 수동 라벨링이 없습니다.
`approach_flip` 시작 = `target`, `approach_pick` 시작 = `flipped`,
`move`/`place` 중 = `empty`, `place` 런 점수 = `ok`/`bad`.

## 2. 정책 학습

```bash
cd /workspace/m.ax/thirdparty/lerobot
STAGE=4 ./train_xarm7_mvla.sh
```

단계는 누적이며, **각 추가분의 효과를 귀속시키기 위해** 존재합니다.
1단계는 SmolVLA를 재현해야 하고, 이후 단계는 한 번에 하나씩만 바꿉니다.

| STAGE | 내용 |
|---|---|
| 1 | 베이스라인 — 헤드 전부 off, 프롬프트 원문. **SmolVLA와 일치해야 함** |
| 2 | + subtask 프롬프트·헤드 |
| 3 | + quality/speed/mistake metadata, 중립 프롬프트 |
| 4 | + value·status 헤드, 경과시간 (기본값) |
| 5 | + subgoal 이미지 |
| 7 | + advantage 조건화 (value 학습 후) |

가중치는 `--policy.type=mvla --policy.pretrained_path=lerobot/smolvla_base`로
로드합니다 — 정책 클래스는 `type`이 정하므로 SmolVLA 가중치를 받고
새 헤드만 초기화됩니다.

### 데이터 어댑터

`data/adapter.py`가 `LeRobotDataset`을 감싸 배치에 다음을 실어줍니다:

| 키 | 출처 |
|---|---|
| `task` | 조립된 프롬프트 (중립/원문 선택 + dropout) |
| `subtask_index`, `value_subtask`, `value_episode`, `status` | `data/derive.py` |
| `elapsed_subtask` | 별도 키. 정규화 **이후** 정책이 state에 붙입니다 |

`datasets/factory.py`가 `policy.type == "mvla"`일 때만 래핑하며,
**헤드가 전부 꺼져 있으면 원본 데이터셋을 그대로 반환**합니다 —
1단계 베이스라인이 진짜 베이스라인이 되도록.

⚠️ `elapsed_subtask`를 데이터셋에서 `observation.state`에 직접 붙이면 안 됩니다.
정규화 통계가 8채널 기준이라 9번째 채널에서 broadcast가 깨집니다.

---

## 3. 데이터 대조 테스트

규칙·파생 타깃이 실제 시연과 어긋나지 않는지 검사합니다. 데이터 파이프라인이나
플래너 규칙을 건드린 뒤 반드시 실행하세요.

```bash
python -m lerobot.policies.mvla.tests.test_against_dataset \
  --dataset /workspace/m.ax/datasets/lerobot/xarm7_kitting_260923
```

| 검사 | 기준 |
|---|---|
| 플래너 ↔ 시연 시퀀스 일치 | ≥ 99% (현재 216/217) |
| 중립 프롬프트 누수 | ≤ 60% (현재 48.7%, 기존 99.6%) |
| 파생 타깃 정합성 | 런 경계·value·status·elapsed 8개 항목 |

---

## 평가 프로토콜 — 반드시 지킬 것

개발 중 실제로 사고가 났던 항목들입니다.

| 규칙 | 지키지 않으면 |
|---|---|
| **플라시보 대조** — 물체가 없는 영역으로 같은 실험 | 배경 단서로 98% 찍고 속음 (실측) |
| **연속 블록 CV** — 에피소드 순서를 **섞지 말 것** | 세션 누수. 섞으면 98%, 안 섞으면 45% |
| **런 단위 집계** — 에피소드 아님 | 32개 에피소드에 실패+재시도 공존 |
| **배경 마스킹** (`mat`) | 무엇을 보고 맞히는지 통제 불가 |

`recognizer/train.py`는 플라시보 마진이 0.15 미만이면 **실패로 종료**합니다.

---

## 측정 결과 (xarm7_kitting_260923)

| | 정확도 | AUC | 플라시보 AUC | 마진 |
|---|---|---|---|---|
| `mat_state` (922 런, 3-way) | 94.5% | 0.991 | 0.810 | +0.181 |
| └ 매트 위 2-class만 (446 런) | 94.6% | 0.983 | **0.555** | +0.428 |
| `tray_placement` (255 런, bad 37) | 95.7% | 0.966 | 0.711 | +0.255 |

`mat_state`의 플라시보가 0.810으로 높은 것은 누수가 아니라 `empty` 클래스 때문입니다 —
물체가 그리퍼에 있으면 팔 위치가 어느 크롭에서든 보입니다. 매트 위 2-class로
한정하면 0.555로 떨어지며, 이 값이 물체 자체를 읽고 있는지에 대한 정직한 지표입니다.
`train.py`가 두 수치를 모두 출력합니다.

`tray_placement` 운영 지점: 임계값 0.3에서 recall 0.92 / precision 0.76.
오경보 1회 = 재파지 1회(~20초), 놓친 불량 = 출하.

참고 — VLM zero-shot(참조 이미지 제공, 블라인드)은 flip **35%**, place **75%**.

---

## 설계 근거

- **인식기가 정책과 분리된 이유**: 자기 평가 헤드는 자기 맹점을 공유합니다.
  SmolVLA 타워 공유도 측정했으나 타워가 86.4M(DINOv2와 동일 크기)이라
  0.3Hz에서 절감이 없고, `mat` 정확도는 오히려 하락, 플라시보는 2배 오염됐습니다.
- **value가 distributional인 이유**: 런 초반엔 성공/실패가 구분 불가하므로
  정답이 이봉 분포입니다. 스칼라 회귀는 두 모드를 평균내 버립니다.
- **분류기가 아니라 value로 실패를 탐지하는 이유**: `place`는 눈에 보이는
  잘못된 결과지만 `flip`은 중간에 정상으로 보이고 "진척이 오지 않음"으로만
  나타납니다. 사후 라벨로 학습한 프레임 분류기는 시도 첫 프레임부터 발화합니다.
- **중립 프롬프트가 필수인 이유**: 기존 프롬프트가 flip 필요 여부를
  99.6% 누설합니다. 그대로 학습하면 이미지를 볼 이유가 없습니다.
- **경과시간을 state에 넣는 이유**: "공중에 든 부품" 한 프레임은 3초와 15초가
  동일합니다. SmolVLA는 history가 없어 정체를 탐지할 수 없습니다.
