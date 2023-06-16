#!/bin/bash -ev
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Run this script at project root by "./linter.sh" before you commit.

{
  black --version | grep -E "23.3.0" > /dev/null
} || {
  echo "Linter requires 'black==23.3.0' !"
  exit 1
}

echo "Running isort..."
isort --sp . .

echo "Running black..."
black -l 100 .

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
