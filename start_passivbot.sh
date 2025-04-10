#!/bin/bash

SESSION_NAME="passivbot_opt"

# Start tmux session if not already running
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION_NAME \
        '/home/myusuf/Projects/passivbot/.venv/bin/python3 /home/myusuf/Projects/passivbot/src/optimize.py /home/myusuf/Projects/passivbot/configs/optimize.json'
fi

