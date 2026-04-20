# Brief: Prompt Augmentation for LeRobot VLA Training

## 배경 & 사용자 의도

- **문제**: `moonjongsul/manufacturing_kitting_dataset` (LeRobot v3.0, Franka FR3, 106 episodes, 43,897 frames)에 instruction이 **"flip object" 하나뿐**. VLA 모델(π0.5, SmolVLA 등)이 이 특정 토큰 시퀀스에 과적합되어 instruction이 사실상 task ID처럼 작동할 위험.
- **목표**: Prompt augmentation으로 language grounding 일반화 확보
  - 동사 변형: flip / rotate / turn over / invert / upside down
  - 목적어 변형: object / part / black part / auto part
  - 표현 조합: "pick and flip object", "turn the part upside down" 등

## 왜 효과가 있는가

VLA 모델은 학습 시 instruction을 VLM backbone에 통과시켜 language embedding을 얻고, 이걸 action decoder의 conditioning으로 사용한다. 같은 행동에 대해 하나의 prompt만 보면 모델이 **"이 특정 토큰 시퀀스 → 이 행동"** 으로 과적합될 위험이 있어 instruction이 사실상 task ID처럼 작동한다.

Paraphrase augmentation을 하면:

- VLM의 semantic space에서 유사한 임베딩들이 모두 같은 행동에 매핑되어, language → action 매핑이 임베딩 기반이 된다.
- 평가 시 약간 다른 문구("pick and flip it", "turn it over")에도 robust해진다.
- 나중에 multi-task로 확장할 때(다른 action과 섞을 때) language grounding이 실제로 구별 역할을 하게 된다.

## 확인된 데이터셋 구조

로컬 캐시: `~/.cache/huggingface/lerobot/moonjongsul/manufacturing_kitting_dataset/`

- **포맷**: LeRobot **v3.0**
- **핵심 발견**: v3.0 스키마가 **에피소드당 여러 task label을 원래 지원**
  - `meta/tasks.parquet`: 현재 `"flip object" → task_index=0` 하나
  - `meta/episodes/chunk-000/file-00X.parquet`: episode row마다 `tasks` 컬럼이 **array 타입** (`array(['flip object'], dtype=object)`) — 이미 복수 instruction 담을 수 있게 설계됨
  - `data/chunk-XXX/file-XXX.parquet`: frame-level `task_index` (int) 컬럼

### info.json 주요 정보
```json
{
  "codebase_version": "v3.0",
  "robot_type": "franka_fr3",
  "total_episodes": 106,
  "total_frames": 43897,
  "total_tasks": 1,
  "fps": 30
}
```

## Paraphrase Pool 설계 원칙

### 원칙 1: Semantic drift를 일으키지 말 것
- "rotate object"는 임의 각도 회전을 포함하는 더 넓은 개념 → 나중에 "rotate 90 degrees" 같은 다른 task와 충돌 가능
- 더 안전한 대안: "rotate object upside down", "flip object over"
- "upside down object" → 명사구라서 어색 → "turn object upside down"

### 원칙 2: 물체 지칭의 일관성
- 장면에 실제 존재하는 물체 기준으로 지칭 선택
- 검정색 파트만 있는 세팅 → "flip black part" OK
- 여러 색이 섞이면 색 지칭 제외

### 원칙 3: VLM의 training distribution 고려
- SmolVLA backbone(SmolVLM)이나 π0.5의 PaliGemma는 일반적인 영어 instruction에 익숙
- 너무 어색한 문구("object flip execute")는 오히려 방해
- 자연스러운 문장이 좋음

### 원칙 4: 수량과 균형
- Task당 8~15개가 sweet spot
- 너무 적으면 augmentation 효과 약함, 너무 많으면 희귀한 phrasing이 under-sample됨

### 제안된 Pool (flip_object task)
```python
# 안전한 paraphrase (의미 일치)
"flip object"
"flip the object"
"flip part"
"flip the part"
"flip the component"
"pick and flip object"
"pick up and flip the part"
"turn object upside down"
"turn the part upside down"
"turn the part over"
"invert the part"
"flip object over"

# 물체별 지칭 (장면에 해당 물체가 있을 때만)
"flip black part"       # 검정색 파트가 있을 때
"flip auto part"        # 자동차 부품 컨텍스트일 때

# 사용자 판단: drift 위험 있어도 일반화 목적으로 포함 검토
"rotate object"
"upside down object"
```

**사용자 방향**: "의미만 같으면 표현 다양화"에 더 무게를 둠 (drift 위험도 일정 부분 감수).

## 구현 옵션

### 접근 A — 데이터셋 자체를 확장 (추천, 영구적)
- `tasks.parquet`에 paraphrase들을 모두 task로 추가 (`task_index` 0~N)
- 각 에피소드의 `tasks` array를 paraphrase 리스트로 확장
- frame-level `task_index`는 에피소드별 단일 index 유지하거나 frame별로 랜덤 샘플링
- 학습 시 LeRobot이 자연스럽게 episode의 `tasks` 리스트에서 하나를 고르게 함

**장점**:
- LeRobot 기존 파이프라인 그대로 활용 (별도 hook 불필요)
- Policy 코드 수정 불필요
- Hub re-push 시 다른 학습 실험에서도 그대로 혜택

### 접근 B — DataLoader/Policy 단계 on-the-fly
- 데이터셋은 그대로 두고, 학습 script에서 batch의 `task` 문자열을 런타임에 paraphrase pool에서 교체
- 실험하기 빠름, ablation에 좋음
- 단: Hub에 공유될 때는 효과 없음

## 검증 방법 (Ablation)

**학습**: `no_aug` vs `aug` 두 weight를 같은 step까지 학습

**평가**: 세 가지 instruction으로 각각 rollout
- Seen: `"flip object"` (학습에 사용된 정확한 원본)
- Seen paraphrase: `"flip the part"` (pool 안에 있던 것)
- Unseen paraphrase: `"reverse the orientation of the part"` (pool에 없던 것)

Augmentation이 효과적이라면 seen paraphrase와 unseen paraphrase에서 `no_aug` 대비 성능 격차가 줄어들어야 함.

## 고려할 점

- Goal-image conditioning 방향으로 가면 prompt augmentation 우선순위가 낮아질 수 있음 (Manufacturing 세팅에서는 언어보다 목표 이미지 지시가 더 자연스러운 interface)
- Language conditioning을 유지하면서 일반화만 넓히려는 목적이면 paraphrase augmentation이 정답에 가까움

## 사용자의 최신 방향

- **향후 task 종류 추가 예정** (flip 외 다른 조작도)
- **여러 label을 미리 작성해두는 방식 선호** → 접근 A 방향
- Pool을 task별로 관리할 구조 필요
- IDE에서 `src/lerobot/datasets/dataset_tools.py` 열어둠 (기존 tooling에 맞춰 구현 의도 추정)

## 다음 단계

1. `dataset_tools.py`의 기존 유틸 확인 → 거기에 맞는 형태로 paraphrase 주입 스크립트 설계
2. Paraphrase pool을 JSON/YAML 설정 파일로 외부화 (task 추가 시 scalable하게)
3. `tasks.parquet`, episode parquet 수정 스크립트 구현
4. 로컬 스모크 테스트 후 Hub에 re-push