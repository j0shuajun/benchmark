# benchmark

## lm_eval CLI

Run benchmarks against an LLM server:

```bash
./lm_eval \
    --runlist spec gmit fdryqa \
    --url http://localhost:8000/v1/chat/completions \
    --model my-model \
    --max_concurrency 4 \
    --max_retries 3 \
    --api_key YOUR_KEY \
    --cache .cache
```

Benchmark data should be placed under `data/<benchmark>/<subject>.csv` and results are written to the `results` directory.
