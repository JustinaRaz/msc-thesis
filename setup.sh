#!/usr/bin/env bash

set -e

sudo apt update

sudo apt install -y \
    libhunspell-dev \
    hunspell \
    hunspell-lt \
    just

uv sync