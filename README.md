# Benchmark

## `lm_eval` CLI

Run benchmarks against an LLM server:

```bash
python -m lm_eval \
    --runlist spec gmit fdryqa \
    --url http://localhost:8000/v1/chat/completions \
    --model my-model \
    --max_concurrency 4 \
    --max_retries 3 \
    --api_key YOUR_KEY
```

- 평가 데이터는 `data/<benchmark>/<subject>.csv` 아래에 배치해야 하며, 결과는 `results`에 저장됩니다.
- `--cache`옵션을 지정하지 않으면 자동으로 캐시가 생성됩니다.
- 서버 불안정 등으로 평가가 중단된 경우, `--cache` 인자로 평가를 재개할 수 있습니다.
- 결과에는 `response`와 `parsed_response`가 저장됩니다.