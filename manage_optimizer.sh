#!/bin/bash
NAME=myscript
DUMP_DIR=~/criu-dumps/$NAME
PYTHON_EXEC="/home/myusuf/Projects/passivbot/.venv/bin/python3"
SCRIPT_PATH="/home/myusuf/Projects/passivbot/src/optimize.py"
CONFIG_PATH="/home/myusuf/Projects/passivbot/configs/optimize.json"
WORKING_DIR="/home/myusuf/Projects/passivbot"  # Add this line to set the working directory

if [ "$1" = "start" ]; then
    echo "Starting script in tmux session '$NAME'..."

    # Check if tmux session already exists
    tmux has-session -t $NAME 2>/dev/null
    if [ $? != 0 ]; then
        echo "Session $NAME does not exist. Creating a new one..."
        tmux new-session -d -s $NAME "cd $WORKING_DIR && $PYTHON_EXEC $SCRIPT_PATH $CONFIG_PATH"
    else
        echo "Session $NAME already exists. Attaching..."
    fi

    # Check if tmux session is running
    tmux ls | grep -q $NAME
    if [ $? -eq 0 ]; then
        echo "Session '$NAME' started successfully. You can attach with: tmux attach -t $NAME"
    else
        echo "Failed to start tmux session. Check logs for errors."
    fi
elif [ "$1" = "save" ]; then
    # Get PID(s) of the running Python script (find parent process)
    PARENT_PID=$(pgrep -f "$SCRIPT_PATH" | head -n 1)

    if [ -z "$PARENT_PID" ]; then
        echo "Could not find running script. Is it started?"
        exit 1
    fi

    # Check if the parent process is valid and running
    if ! ps -p $PARENT_PID > /dev/null; then
        echo "The parent process with PID $PARENT_PID is not running."
        exit 1
    fi

    # Create dump directory if not exists
    mkdir -p $DUMP_DIR

    # Run criu dump (dump the entire process tree)
    echo "Attempting to dump the entire process tree of PID $PARENT_PID..."
    sudo criu dump -t $PARENT_PID -D $DUMP_DIR --tree $PARENT_PID

    if [ $? -eq 0 ]; then
        echo "Script and its processes dumped successfully to $DUMP_DIR."
    else
        echo "Failed to dump the script process and its tree. Please check the CRIU logs for errors."
    fi
elif [ "$1" = "restore" ]; then
    echo "Restoring script from dump..."
    tmux new-session -d -s ${NAME}-restore "sudo criu restore -D $DUMP_DIR --shell-job --tcp-established"
    echo "Restoration attempted in tmux session '${NAME}-restore'."
    echo "You can check logs in $DUMP_DIR/restore.log"
else
    echo "Usage: $0 start|save|restore"
    exit 1
fi

