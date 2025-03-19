#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

python run/baseline_uncertainty.py \
    --data_path=data/ \
    --model_name_hf=meta-llama/Llama-2-7b-chat-hf \
    --task_type=controlled_ver \
    --hf_use_auth_token PERSONAL_TOKEN
