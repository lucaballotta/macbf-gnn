#!/bin/bash

LOG_PATH=$1
N_EPI=$2
N_AGENT=$3
VIDEO=$4

# mkdir for videos
if [ ! -d "$LOG_PATH/videos" ]; then
  mkdir "$LOG_PATH/videos"
fi

# mkdir for logs
if [ -f "$LOG_PATH/test_log.csv" ]; then
  rm "$LOG_PATH/test_log.csv"
fi
touch "$LOG_PATH/test_log.csv"

# test the agent
if [ "$VIDEO" == 1 ]; then
  MAX_CORE=$((5))
else
  MAX_CORE=$((8))
fi
I_EPI=$((0))
while [ $I_EPI -lt $((N_EPI)) ]; do
  for ((i=0; i<MAX_CORE; ++i)); do
    if [ "$VIDEO" == 1 ]; then
      python "test.py" --path "$LOG_PATH" --epi "1" -n "$N_AGENT" --seed "$I_EPI"&
    else
      python "test.py" --path "$LOG_PATH" --epi "1" -n "$N_AGENT" --seed "$I_EPI" --no-video&
    fi
    I_EPI=$((I_EPI+1))
    if [ $I_EPI == $((N_EPI)) ]; then
      break
    fi
  done
  wait
done
