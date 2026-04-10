#!/bin/bash
# record_demo.sh
# 사용법:
#   ./record_demo.sh                        # default: task="task", 번호 자동
#   ./record_demo.sh pick_place             # task만 지정, 번호 자동
#   ./record_demo.sh pick_place 005         # 둘 다 직접 지정

set -e


TASK_NAME=${1:-"task"}
BAG_ROOT="./rosbag"
TASK_DIR="${BAG_ROOT}/${TASK_NAME}"

# ── episode 번호 자동 증가 ──────────────────────
if [ -z "$2" ]; then
    # 기존 episode_xxx 폴더 중 최대값 + 1
    LAST=$(ls -d "${TASK_DIR}"/episode_* 2>/dev/null \
        | grep -oE '[0-9]+$' \
        | sort -n | tail -1)

    if [ -z "$LAST" ]; then
        EPISODE_NUM="000"
    else
        EPISODE_NUM=$(printf "%03d" $(( 10#${LAST} + 1 )))
    fi
else
    EPISODE_NUM=$(printf "%03d" "$2")
fi

OUTPUT_DIR="${TASK_DIR}/episode_${EPISODE_NUM}"

# ── 중복 방지 ───────────────────────────────────
if [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️  이미 존재: $OUTPUT_DIR"
    read -p "덮어쓰시겠습니까? [y/N] " yn
    [[ "$yn" != "y" ]] && exit 0
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$TASK_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Task    : $TASK_NAME"
echo " Episode : $EPISODE_NUM"
echo " Output  : $OUTPUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Ctrl+C 로 녹화 종료"
echo ""

ROS_DOMAIN_ID=${CAM_DOMAIN_ID} ros2 bag record \
    -o "${OUTPUT_DIR}" \
    --max-bag-size 2147483648 \
    /gello/joint_states \
    /gripper/gripper_client/target_gripper_width_percent \
    /joint_states \
    /franka_robot_state_broadcaster/current_pose \
    /dxl_parallel_gripper/joint_states \
    /front_view/color/image_raw/compressed \
    /side_view/color/image_raw/compressed \
    /wrist/front/color/image_raw/compressed \
    /wrist/rear/color/image_raw/compressed

echo ""
echo "✅ 저장 완료: $OUTPUT_DIR"
ros2 bag info "$OUTPUT_DIR"
