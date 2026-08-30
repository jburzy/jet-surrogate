#!/bin/bash
ps -o pid,ppid,stat,pcpu,rss,etime,cmd -u jburzyns | grep -v 'ps -o\|grep\|inspect.sh' | head -20
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
for py in $(pgrep -u jburzyns -f train_surrogate); do echo "PID $py wchan=$(cat /proc/$py/wchan) threads=$(ls /proc/$py/task | wc -l) state=$(grep State /proc/$py/status)"; done
