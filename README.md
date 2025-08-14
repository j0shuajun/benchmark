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
    --api_key YOUR_KEY
```

Benchmark data should be placed under `data/<benchmark>/<subject>.csv` and results are written to the `results` directory.

Cache files are written to `.cache/<benchmark>_<model>_<time>.json` with slashes converted to underscores; a new cache is created automatically unless `--cache` is provided to resume from an existing file. Result CSVs include both the raw `Response` and the extracted `parsed_response`. When evaluating responses, the runner extracts the first multiple-choice option from the model output using a comprehensive regular expression before scoring.
