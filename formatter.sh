#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR=$(dirname "$0")

pushd "$SCRIPT_DIR"

black . --target-version=py311 --line-length=120 --exclude=thirdparty/tiny-cuda-nn
isort . --skip=thirdparty/tiny-cuda-nn --profile=black

popd
