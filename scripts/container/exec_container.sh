#!/usr/bin/env bash
#
# Run commands in a running container.
# Without any arguments to keep interactive.

set -e

# Use argument $1 as $NAME.
NAME=${1:-${NAME}}

# The directory contains current script.
DIR=$(dirname $(realpath "$BASH_SOURCE"))
# The directory contains .git directory.
REPO_DIR=${REPO_DIR:-$(
  d=$DIR
  while [[ $d =~ ^$HOME ]]; do
    [[ -d $d/.git ]] && echo $d && break
    d=$(dirname "$d")
  done
)}
[[ -n $REPO_DIR ]] || (
  echo >&2 "Failed to find working directory"
  exit 1
)

# Set default $NAME.
if [[ -z $NAME || $NAME == "--" ]]; then
  NAME=$(basename $REPO_DIR)
fi

WORK_DIR_IN_CONTAINER=${WORK_DIR_IN_CONTAINER:-/$(basename $REPO_DIR)}

# Check if the container exists.
if ! docker ps -a --format "{{.Names}}" | grep -q "^$NAME$"; then
  echo >&2 "Container [$NAME] does not exist"
  exit 1
fi
# Check if the container is running.
if ! docker ps --format "{{.Names}}" | grep -q "^$NAME$"; then
  echo "Starting container [$NAME] ..."
  docker start $NAME >/dev/null
fi

echo "Execute container [$NAME] ..."

# Allow docker to connect to the X server.
xhost +local:docker &>/dev/null || true

function cleanup() {
  # NOTE: Kill `docker exec` command will not terminate the spawned process.
  # Check https://github.com/moby/moby/issues/9098.

  # Escape the command string.
  local cmd=$(echo "${*}" | sed 's/[]\/$*+.^[]/\\&/g')
  # Kill if still running.
  if (($#)) && docker exec $NAME bash -c "pgrep -af '$cmd'"; then
    echo "Kill running processes by pattern: $cmd"
    docker exec $NAME bash -c "pkill -ef '$cmd'"
  fi

  # Disallow docker to connect to the X server.
  xhost -local:docker &>/dev/null || true
}
trap "cleanup ${*}" EXIT

echo $DOCKER_USER

if (($#)); then
  echo "Executing bash command '${*}' inside container [$NAME] ..."
  docker exec -t \
    -u $USER \
    -e USER=$USER \
    -e HISTFILE=$WORK_DIR_IN_CONTAINER/.${NAME}_bash_history \
    $NAME \
    /bin/bash -ic "${*}"
else
  echo "Executing an interactive bash shell inside container [$NAME] ..."
  docker exec -it \
    -u $USER \
    -e USER=$USER \
    -e HISTFILE=$WORK_DIR_IN_CONTAINER/.${NAME}_bash_history \
    $NAME \
    /bin/bash
fi
