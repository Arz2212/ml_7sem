#!/usr/bin/env bash
# запуск обоих заданий
set -e
cd "$(dirname "$0")"
mkdir -p results
.venv/bin/python points_task.py > results/points_output.txt
.venv/bin/python text_task.py   > results/text_output.txt
echo "готово: results/*.png, results/*_output.txt"
