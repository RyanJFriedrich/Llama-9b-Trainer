"""Head-to-head prefill probe: spawn llama-server, harvest prompt_probabilities,
ALWAYS kill the server on exit (success, exception, or Ctrl+C).

Usage:
  headtohead_probe.py <model.gguf> <gpu_layers> <out.json> [runs] [ctx_size]

Reads token ids from data_pipeline/wiki_4000_tokens.json.
"""
import json
import subprocess
import sys
import time
import urllib.request

MODEL = sys.argv[1]
GPU_LAYERS = sys.argv[2]
OUT = sys.argv[3]
RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
CTX = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
PORT = 8390
BASE = f'http://127.0.0.1:{PORT}'

SERVER_CMD = [
    'LlamaCPPBinaries/llama-server.exe',
    '--model', MODEL,
    '--ctx-size', str(CTX),
    '--fit', 'off',
    '--gpu-layers', str(GPU_LAYERS),
    '-ctk', 'bf16', '-ctv', 'bf16',
    '--cache-ram', '0',
    '--checkpoint-every-n-tokens', '-1',
    '--flash-attn', 'on',
    '--batch-size', '2048', '--ubatch-size', '512',
    '--parallel', '1',
    '--no-webui',
    '--host', '127.0.0.1', '--port', str(PORT),
]

def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)

def wait_healthy(deadline_s=600):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            with urllib.request.urlopen(BASE + '/health', timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False

server = None
server_log = open('data_pipeline/server_probe.log', 'wb')
try:
    print(f'starting server: {MODEL} (gpu_layers={GPU_LAYERS})', flush=True)
    server = subprocess.Popen(SERVER_CMD, stdout=server_log, stderr=subprocess.STDOUT)
    if not wait_healthy():
        raise RuntimeError('server did not become healthy; see data_pipeline/server_probe.log')
    print('server healthy', flush=True)

    ids = json.load(open('data_pipeline/wiki_4000_tokens.json'))
    body = {'prompt': ids, 'n_predict': 0, 'n_probs': 10,
            'prompt_logprobs': True, 'stream': False, 'cache_prompt': False}
    for run in range(1, RUNS + 1):
        t0 = time.time()
        d = post('/completion', body)
        wall = time.time() - t0
        if run == RUNS:
            with open(OUT, 'w', encoding='utf-8') as fh:
                json.dump(d, fh)
        t = d['timings']
        print(f"run {run}: prompt_ms={t['prompt_ms']:.0f} "
              f"({t['prompt_per_second']:.1f} tok/s) wall={wall:.2f}s "
              f"pp_len={len(d['prompt_probabilities'])}", flush=True)
finally:
    if server is not None:
        print('killing server', flush=True)
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=15)
        # belt-and-braces: nothing may survive on the port
        subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'],
                       capture_output=True)
    server_log.close()
    print('server down', flush=True)
