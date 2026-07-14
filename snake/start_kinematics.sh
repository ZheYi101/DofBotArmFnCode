#!/usr/bin/env bash
set -u

source /opt/ros/noetic/setup.bash
source /home/dofbot/dofbot_ws/devel/setup.bash

if ! rostopic list >/dev/null 2>&1; then
  echo "错误：ROS Master 不可用。请先启动机器原有 ROS 大程序。" >&2
  exit 1
fi

if rosservice list 2>/dev/null | grep -qx '/dofbot_kinemarics'; then
  echo "dofbot_kinemarics 已运行"
  exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "dofbot_kinemarics 未运行" >&2
  exit 2
fi

echo "启动 dofbot_kinemarics；保持本终端运行，Ctrl-C 可退出。"
exec rosrun dofbot_info dofbot_server
