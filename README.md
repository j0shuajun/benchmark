# Benchmark

## `lm_eval` CLI

Run benchmarks against an LLM server:

```bash
python -m lm_eval \
    --runlist "spec gmit foundryqa" \
    --url https://<your.api.net>/v1/chat/completions \
    --model <your_model> \
    --max_concurrency <max_concurrency> \
    --max_retries <max_retries> \
    --api_key <your_api_key> \
    --cache <path_to_cache_file>
```

- 평가 데이터는 `data/<benchmark>/<subject>.csv` 아래에 배치해야 하며, 결과는 `results`에 저장됩니다.
- 서버 불안정 등으로 평가가 중단된 경우, `--cache`에 캐시 파일 경로를 지정하면 해당 파일을 불러와 이어서 사용할 수 있습니다.
- `--cache`를 지정하지 않으면 `.cache/` 아래에 `<model>_<timestamp>.json` 파일이 새로 생성됩니다.
- 결과에는 `response`와 `parsed_response`가 저장됩니다.
