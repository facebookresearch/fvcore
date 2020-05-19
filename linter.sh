#!/bin/bash -ev
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Run this script at project root by "./linter.sh" before you commit.

{
  black --version | grep -E "(19.3b0.*6733274)|(19.3b0\\+8)" > /dev/null
} || {
	echo "Linter requires 'black @ git+https://github.com/psf/black@673327449f86fce558adde153bb6cbe54bfebad2' !"
	exit 1
}

echo "Running isort..."
isort -y -sp .

echo "Running black..."
black .

echo "Running flake8..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  python3 -m flake8 .
fi

command -v arc > /dev/null && {
  echo "Running arc lint ..."
  arc lint
}
