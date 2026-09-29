# ros_inference — xarm7 kitting rollout

SmolVLA / MVLA 체크포인트로 xArm7을 EEF 속도(twist) 제어한다. 관측·액션
정의는 `m.ax/ros2/src/max_data_collect`가 녹화한 것과 같다.

```
카메라 x3 ─┐                                   ┌─> /xarm/vc_set_cartesian_velocity (mm/s, rad/s)
eef_pose  ─┼─> node.py ─> RolloutController ───┼─> 그리퍼 Float32 (바뀔 때만)
eef_twist ─┤              │  30Hz tick          ├─> /data_collection/rollout_action (TeleopAction, 레코더용)
gripper   ─┘              └─ worker 스레드       └─> /ros_inference/status (JSON)
                             (정책 추론)
```

| 파일 | 역할 | ROS 필요 |
|---|---|---|
| `config/xarm7_kitting.yaml` | 토픽·단위·안전 제한·체크포인트 | |
| `observation.py` | 센서값 → 정책 입력 (frame_builder와 같은 변환) | |
| `action.py` | 정책 출력 → 명령, **안전 제한 전부** | |
| `policy_runner.py` | 체크포인트 로드, 청크 추론, MVLA 헤드 읽기 | |
| `subtask.py` | MVLA: 헤드·인식기 → 플래너(TaskManager) → 프롬프트, 작업 종료 판단 | |
| `recognition.py` | 상태 인식기 실행 (시작 상태, flip·place 검증), 별도 스레드 | |
| `controller.py` | 시간 기준 청크 실행, 추론 스케줄링, 상태 머신 | |
| `replay.py` | 녹화 에피소드로 전체 파이프라인 오프라인 실행 | |
| `node.py` | ROS2 입출력만 | ✔ |

제어 로직은 전부 ROS 없이 돈다. 그래서 이 PC에서 `replay.py`와 테스트로
검증했고, 로봇 PC에서는 `node.py`만 새로 도는 코드다.

## 실행

```bash
# 로봇 PC: ROS2 + xarm_ros2 + max_interfaces 워크스페이스를 source 한 뒤
source /opt/ros/humble/setup.bash
source ~/workspace/m.ax/ros2/install/setup.bash
cd ~/workspace/lerobot && source .venv/bin/activate   # lerobot이 설치된 환경

python -m ros_inference.node --config ros_inference/config/xarm7_kitting.yaml \
    --checkpoint <ckpt>/checkpoints/050000 --speed-scale 0.3

ros2 service call /ros_inference/start std_srvs/srv/Trigger
ros2 service call /ros_inference/stop  std_srvs/srv/Trigger
ros2 topic echo /ros_inference/status
ros2 topic pub --once /ros_inference/task std_msgs/String "data: pick"   # 다음 start의 과제
```

`.venv`의 파이썬이 ROS의 `rclpy`를 import할 수 있어야 한다 (파이썬 버전 일치,
`PYTHONPATH`에 ROS site-packages). 안 맞으면 추론을 별도 프로세스로 나누는
방법이 있다 — 필요하면 말할 것.

### 오프라인 검증 (이 PC, ROS 없음)

```bash
python -m pytest ros_inference/tests -q
python -m ros_inference.replay --config ros_inference/config/xarm7_kitting.yaml \
    --checkpoint <ckpt> --episode 33
```

replay는 **개루프**다(로봇이 명령대로 안 움직임). 단위·카메라 매핑·청크
타이밍·안전 제한이 맞는지를 본다. val 에피소드 33 결과(50k 체크포인트):

| | MVLA s4 | SmolVLA |
|---|---|---|
| 정책 vs 녹화 액션 상관 (vx vy vz wx wy wz) | .77 .64 .91 .92 .96 .64 | .77 .59 .93 .91 .96 .70 |
| 첫 액션까지 | 0.87 s | 0.77 s |
| MVLA 시작 상태 판단 | flip 필요 (정답) | — |
| subtask 추적 정확도 | 91% | — |

## 녹화 연동 (rollout 데이터 쌓기)

`auto_record: true`면 start 때 레코더 START, 끝날 때 SAVE를 부른다(실패한
rollout도 저장 — 다음 학습 데이터이자 advantage 추정의 재료다). 레코더는
`demo_data_collect_config.yaml`을 복사해 두 곳만 바꿔 띄운다:

```yaml
source: rollout
action:
  topic: /data_collection/rollout_action   # 이 노드가 퍼블리시
```

`TeleopAction.cartesian_velocity`는 데모 녹화와 같은 **m/s**로 채운다.

## 확인이 필요한 가정 — 첫 실기 전에

| 가정 | 근거 | 틀리면 |
|---|---|---|
| 선속도는 m/s로 녹화됐고 xArm 서비스엔 mm/s로 보낸다 | 데이터셋 action 범위 ±0.25. `TeleopAction.msg` 주석(mm/s)과 다르다 | 1000배 빠르거나 느림 → 첫 실행은 `--speed-scale 0.1`로 |
| 속도는 베이스 좌표(`is_tool_coord=False`) | `is_tool_coord`가 녹화되지 않았다 | 시연 중 툴 좌표를 썼다면 그 구간 액션의 의미가 다름 |
| 명령 경로 = `vc_set_cartesian_velocity` + mode 5 | eef_state_publisher 주석 ("teleop drives the arm with vc_set_cartesian_velocity") | teleop 노드 코드가 이 저장소에 없어 서비스 이름·타입은 xarm_ros2 표준으로 가정 |
| 그리퍼 = `/gripper/gripper_client/target_gripper_width_percent` Float32 0/1 | max_server config, TeleopAction 주석 | |
| 카메라 토픽·해상도 640x480, 회전 0 | demo_data_collect_config.yaml | 이미지 분포가 달라지면 조용히 성능 저하 |
| 작업공간 박스 | 녹화 eef 위치 0.1–99.9 퍼센타일 | **z 하한은 셀에서 실측** |

xarm_api에서 `vc_set_cartesian_velocity` 서비스가 켜져 있어야 한다
(xarm_ros2의 `xarm_params`에서 서비스 활성화). teleop이 쓰고 있으므로 켜져 있을 것이다.

## 안전 장치

- `speed_scale` — 모든 속도에 곱함. 첫 실기는 0.3 이하로.
- 축별 상한(데이터셋 최대값), 틱 간 변화율 제한(청크 전환 튐 방지)
- 작업공간 박스 — 밖으로 향하는 성분만 0 (안쪽 복귀는 허용)
- 관측이 0.3초 이상 끊기면 정지 (에피소드는 유지, 돌아오면 재개)
- 모든 속도 명령에 `duration=0.2s` — 노드가 죽거나 멈춰도 팔이 스스로 멈춤
- 에피소드 시간 상한, MVLA 매니저의 재시도 상한 → 초과 시 정지
- **비상정지는 하드웨어 버튼이 1순위.** 이 노드는 그 대체가 아니다.

## MVLA 런타임 동작

세 개의 시계: 정책 30Hz / 헤드 ~5Hz (전진, value) / **인식기 — 경계에서만** (확정 판정)

1. **시작**: 그리퍼 effort로 파지 여부. 아니면 인식기 mat 판정 3프레임(0.1/0.4/0.7초)
   → `target`/`flipped` → 플래너가 **메인 prompt(목표)**로 plan을 세운다.
   `empty`·`abstain`이면 정지(사람 필요). 인식기가 없으면 헤드의 approach_flip/pick 확률로 대체
2. 정책 프롬프트 = `Task: <중립 목표>. Subtask: <plan의 현재>. Quality: 5. Mistake: false.`
3. **전진**: 헤드가 plan의 다음 subtask를 2번 연속 예측 (헤드는 정책과 같은 프롬프트로 읽음)
4. **flip 검증**: flip에서 넘어간 직후 +3/12/21프레임에 mat 판정. `flipped`가 아니면(2회)
   approach_flip부터 재계획. 판정 동안 정책은 이미 approach_pick으로 움직인다
5. **place 검증**: 그리퍼가 실제로 열린 순간 +8/18/28프레임에 tray 판정.
   `ok` 2회 → 작업 완료, `bad` 2회 → **트레이 재시도** `approach_pick → pick → place`
   (데이터에서 place 실패 뒤 작업자는 항상 move 없이 트레이에서 다시 집었다)
6. 실패: 검증 실패 또는 매니저 정체 규칙(value ≤ -0.95가 6초 — val에서 거의 안 걸림)
   → 재계획. 재시도 상한(subtask별 3, 전체 6) 초과 → 정지

캡처 시각은 인식기 학습 샘플의 시점과 같다(`recognizer/dataset.py`). 다르면 인식기가
학습 때와 다른 장면을 본다.

### 메인 prompt 작업 종료

**plan의 마지막 subtask가 끝나면 메인 prompt의 작업이 끝난 것이다.** 목표가
plan을 정하므로(pick → `…pick`에서 끝, kit → `…place`에서 끝) 목표가 바뀌면
끝나는 지점도 바뀐다. 마지막 subtask의 완료는:

| 마지막 subtask | 완료 판단 | `basis` |
|---|---|---|
| place (kit 목표) | 인식기 tray `ok` 2회 | `recognizer_tray` |
| place인데 릴리스가 안 보임 | value 조건이 5초 유지되면 검증 없이 | `unverified` |
| pick (pick 목표), 인식기 없음 | `value_subtask ≥ -0.05` 또는 `value_episode ≥ -0.05` 3회 연속 | `last_subtask_value` |

`value_episode` 하나로만 판단하지 않는 이유 (`eval_termination`, val 58개):

| | 결과 |
|---|---|
| 끝 시점 탐지 (value_episode ≥ -0.05, 3회) | 성공 에피소드 90~95%, 실제 끝보다 ~1.2초 전 |
| 작업 중간 오발동 | 0~2% (마지막 subtask로 제한하면 0%) |
| **prompt만 pick↔kit로 바꿨을 때** | **값이 그대로** (kit 에피소드 pick 끝: -0.383 vs -0.385) |

헤드는 "이 prompt의 작업이 끝났나"가 아니라 장면을 보고 끝을 판단한다. 목표에
따라 종료 지점이 달라지는 건 plan이 보장한다.

종료 시 출력:
- `/ros_inference/task_complete` (latched JSON, 1회):
  `{goal, main_prompt, initial_state, basis, duration_s, subtasks:[{name,t_start,t_end,outcome}], retries, value_subtask, value_episode}`
- `/ros_inference/status`: `status: done`, `task_complete: true`, `completion: {...}`
- 0 속도 명령, 레코더 SAVE, 로그 파일 첫 줄에 `completion`

진행 중 `/ros_inference/status`에는 `goal, main_prompt, subtask, remaining, on_last_subtask,
subtask_elapsed_s, task_elapsed_s, completion_streak, retries, value_subtask(+spread),
value_episode, head_subtask`가 1Hz로 나간다.

replay (정책 50k + 인식기, val):

| 에피소드 | 녹화 | runtime |
|---|---|---|
| 105, 262, 114 | 정상 kit | 시작 판정 정확, flip 검증 통과, tray `ok` → 완료 (끝 0~0.4초 전) |
| 291 | 쥔 채 시작 | move → place → `ok` → 완료 |
| 113, 280, 282 | place 실패 → 트레이 재시도 | tray `bad` → 재시도 → `ok` → 완료 (녹화와 같은 흐름) |
| 249 | place 실패 → 재시도 성공 | 첫 `bad` 정확, **두 번째(실제 성공)도 `bad`로 오판** → 불필요한 재시도 |
| 33 | pick 목표 | 녹화가 들어 올린 직후 끝나 value 3회를 못 채움 — **실기 확인 필요** |

SmolVLA에는 헤드가 없어 종료 판단이 없다 (타임아웃 또는 사람이 정지).

`Speed:` 필드는 기본 생략. 15퍼센타일(500프레임)은 전체 과제 길이
(900–1900)에 비해 분포 밖이다.
