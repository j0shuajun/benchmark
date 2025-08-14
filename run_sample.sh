python -m lm_eval \
    --runlist "spec gmit foundryqa" \
    --url https://<your.api.net>/v1/chat/completions \
    --model <your_model> \
    --max_concurrency <max_concurrency> \
    --max_retries <max_retries> \
    --api_key <your_api_key> \
    --cache <path_to_cache_file>
