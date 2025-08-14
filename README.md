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
- `data/<benchmark>/system_prompt.txt`가 system prompt로 사용됩니다.
- 서버 불안정 등으로 평가가 중단된 경우, `--cache`에 캐시 파일 경로를 지정하면 해당 파일을 불러와 이어서 사용할 수 있습니다.
- `--cache`를 지정하지 않으면 `.cache/` 아래에 `<model>_<timestamp>.json` 파일이 새로 생성됩니다.
- 결과에는 `response`와 `parsed_response`가 저장됩니다.

## Prepare Data
- 평가 데이터는 다음과 같은 형태여야 합니다.
- `RAG_Contexts` 컬럼이 있으면 user prompt에 삽입되고, 없으면 질문과 보기만 사용됩니다.
- `RAG_Contexts`는 chunk들을 `\n\n`으로 join 해서 넣어두면 됩니다.

|question|answer|A|B|C|D|RAG_Contexts|
|-|-|-|-|-|-|-|
|질문|1~4|보기A|보기B|보기C|보기D|검색된 문서들|
