#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Check if the required arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 <HEAD> <MODEL_SIZE>"
  exit 1
fi
HEAD=$1
MODEL_SIZE=$2
curl http://$HEAD:8000/${MODEL_SIZE}/v1/chat/completions -H "Content-Type: application/json" -d '{
      "model": "meta-llama/Meta-Llama-3.1-'${MODEL_SIZE}'-Instruct",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who is CEO of Meta?"}
      ],
      "temperature": 0.7
}'