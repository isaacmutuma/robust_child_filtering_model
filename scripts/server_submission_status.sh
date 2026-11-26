#!/usr/bin/env bash
# Progress of scripts/server_submission_run.sh (aggregate only: counts and log lines, never images).
echo "run:   $(pgrep -f 'nbconvert.*202555404_IsaacMutuma' >/dev/null && echo running || echo 'not running')"
echo "fetch: $( [[ -e ~/iccwd/COMPLETE ]] && echo done || ( [[ -d ~/iccwd/images ]] && echo "in progress, $(ls ~/iccwd/images | wc -l) images" || echo 'no ~/iccwd' ) )"
grep '^==\|^!!\|settings:\|cells with errors' ~/submission_run.log 2>/dev/null | tail -12
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
