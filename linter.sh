#!/bin/bash -ev
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Run this script at project root by "./linter.sh" before you commit.

{
  black --version | grep -E "24.2.0" > /dev/null
} || {
  echo "Linter requires 'black==24.2.0' !"
  exit 1
}

echo "Running isort..."
isort -y -sp .

echo "Running black..."
black .

echo "Running flake8..."
if [ -x "$(command -v flake8)" ]; then
  flake8 .
else
  python3 -m flake8 .
fi

command -v arc > /dev/null && {
  echo "Running arc lint ..."
  arc lint
}
