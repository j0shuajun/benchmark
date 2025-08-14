python -m lm_eval \
    --runlist "spec gmit foundryqa" \
    --url https://ais-dsllm-dev-kai-gpt-oss-120b.aiserving.aip.samsungds.net/v1/responses \
    --model openai/gpt-oss-120b \
    --max_concurrency 30 \
    --max_retries 3 \
    --api_key <your_api_key>