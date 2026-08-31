"""Bulk corpus scorer: stream wiki/open-web text, score 8k chunks with the local
8B GGUF via the fork llama-server (prompt_logprobs), pack to legacy-style NPZ.

NPZ contract (per shard file):
  tokens        u32 [N]         raw token ids (EOS-separated doc stream, chunked)
  teacher_ids   i32 [N,K]       row t predicts tokens[t]. Slot 0 = GT (the true
                                token at position t) with its TRUE teacher prob;
                                slots 1..K-1 = teacher top-K minus GT, shuffled
                                as intact (id, prob) pairs
  teacher_probs f32 [N,K]       true softmax mass, NOT renormalized (rows sum to
                                captured mass <= 1; the tail is whatever is left)
  loss_mask     u8  [N]         0 at position 0 of each chunk (no distribution
                                exists for it), 1 elsewhere
  chunk_start   i64 [C], chunk_length i64 [C]

Modes:
  default        runs forever until Ctrl+C
  --num-chunks N stops after N chunks

Ctrl+C is graceful: in-flight chunks finish, the partial shard is flushed, the
manifest is updated, and the server is killed. Restarting with the same
--out-dir RESUMES: shard numbering continues and the (deterministic, seeded)
corpus stream is fast-forwarded past the chunks already scored.

The llama-server is ALWAYS killed on exit (success, exception, or Ctrl+C).

Run from the repo root:  python data_pipeline/run_bulk_score.py --help
"""
import argparse
import concurrent.futures as cf
import json
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.utils.log import log

DOC_SEP = 128001  # <|end_of_text|> — pretraining-style document separator
PORT = 8390
BASE = f'http://127.0.0.1:{PORT}'
HERE = Path(__file__).resolve().parent

STOP = threading.Event()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='QuantizedModel/meta-llama-3.1-8b-instruct.Q8_0.gguf')
    p.add_argument('--gpu-layers', type=int, default=999)
    p.add_argument('--batch', type=int, default=1,
                   help='concurrent chunks in flight (server slots). NOTE: on this '
                        'fork build, prompt_logprobs post-processing is CPU-bound and '
                        'serializes across slots — batch 2 measured SLOWER (155 vs '
                        '620 tok/s per chunk). Left configurable for experimentation.')
    p.add_argument('--seq-len', type=int, default=8192, help='tokens per chunk')
    p.add_argument('--k', type=int, default=32, help='teacher_ids/probs columns')
    p.add_argument('--num-chunks', type=int, default=None,
                   help='stop after N chunks (default: run forever)')
    p.add_argument('--chunks-per-npz', type=int, default=128,
                   help='chunks packed per output .npz shard')
    p.add_argument('--wiki-frac', type=float, default=0.5,
                   help='fraction of documents drawn from wikimedia/wikipedia '
                        '(remainder: HuggingFaceFW/fineweb-edu sample-10BT)')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--min-doc-chars', type=int, default=200)
    p.add_argument('--out-dir', default='data_pipeline/bulk_out')
    p.add_argument('--tokenizer', default='OriginalModel')
    p.add_argument('--fresh', action='store_true',
                   help='ignore an existing manifest in out-dir and start over')
    return p.parse_args()


# ---------------------------------------------------------------- server ----
def server_cmd(args):
    # this build splits --ctx-size across slots: size it per-slot
    return [
        'LlamaCPPBinaries/llama-server.exe',
        '--model', args.model,
        '--ctx-size', str(args.batch * (args.seq_len + 64)),
        '--fit', 'off',
        '--gpu-layers', str(args.gpu_layers),
        '-ctk', 'bf16', '-ctv', 'bf16',
        '--cache-ram', '0',
        '--checkpoint-every-n-tokens', '-1',
        '--flash-attn', 'on',
        '--batch-size', '2048', '--ubatch-size', '512',
        '--parallel', str(args.batch),
        '--no-webui',
        '--host', '127.0.0.1', '--port', str(PORT),
    ]


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


def score_chunk(ids, k):
    """One prefill; returns prompt_probabilities (len = len(ids) - 1)."""
    body = json.dumps({
        'prompt': ids, 'n_predict': 0, 'n_probs': k,
        'prompt_logprobs': True, 'stream': False, 'cache_prompt': False,
    }).encode('utf-8')
    req = urllib.request.Request(BASE + '/completion', data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.load(resp)
    return d['prompt_probabilities'], d['timings']


# ---------------------------------------------------------------- corpus ----
# TODO(later): prestream blocks of each corpus to local disk instead of live
# streaming. Datasets are deterministic per revision (wikipedia 20231101.en is
# a frozen snapshot), so pin revision=..., materialize N docs per source, and
# track the doc cursor in the manifest — resume becomes instant (no tokenize
# fast-forward) and long runs stop depending on HF availability mid-run.
def doc_streams(args):
    """Seeded interleave of wikipedia + fineweb-edu, streaming."""
    from datasets import load_dataset
    wiki = load_dataset('wikimedia/wikipedia', '20231101.en',
                        split='train', streaming=True)
    fw = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT',
                      split='train', streaming=True)
    it_wiki, it_fw = iter(wiki), iter(fw)
    rng = random.Random(args.seed)
    while True:
        it = it_wiki if rng.random() < args.wiki_frac else it_fw
        try:
            doc = next(it)
        except StopIteration:
            log('a stream ended (should not happen while streaming); restarting it')
            if it is it_wiki:
                it_wiki = iter(load_dataset('wikimedia/wikipedia', '20231101.en',
                                            split='train', streaming=True))
            else:
                it_fw = iter(load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT',
                                          split='train', streaming=True))
            continue
        text = doc.get('text', '')
        if len(text) >= args.min_doc_chars:
            yield text


def chunk_stream(args, tokenizer):
    """Yield lists of seq_len token ids from the EOS-separated doc stream."""
    buf = []
    for text in doc_streams(args):
        buf.extend(tokenizer(text, add_special_tokens=False)['input_ids'])
        buf.append(DOC_SEP)
        while len(buf) >= args.seq_len:
            yield buf[:args.seq_len]
            buf = buf[args.seq_len:]


# ------------------------------------------------------------- targets ------
def build_rows(pp, chunk_len, k, rng):
    """prompt_probabilities -> teacher_ids/teacher_probs rows (pairing intact).

    Row t describes the distribution for tokens[t]. Slot 0 = GT with its true
    prob; remaining slots = teacher top-k minus GT, shuffled as pairs.
    Row 0 is a placeholder (id -1): no distribution exists for the first token.
    """
    ids = np.zeros((chunk_len, k), dtype=np.int32)
    probs = np.zeros((chunk_len, k), dtype=np.float32)
    ids[0, 0] = -1  # position 0 has no distribution; loss_mask=0 there
    for t, e in enumerate(pp, start=1):
        gt, gt_lp = e['id'], e['logprob']
        rest = [(x['id'], x['logprob']) for x in e['top_logprobs'] if x['id'] != gt]
        rest = rest[:k - 1]
        rng.shuffle(rest)
        ids[t, 0] = gt
        probs[t, 0] = np.exp(np.float32(gt_lp))
        for j, (tid, lp) in enumerate(rest, start=1):
            ids[t, j] = tid
            probs[t, j] = np.exp(np.float32(lp))
    return ids, probs


# ------------------------------------------------------------------ main ----
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / 'manifest.json'

    if manifest_path.exists() and not args.fresh:
        manifest = json.loads(manifest_path.read_text())
        skip_chunks = manifest.get('stream_cursor', manifest.get('total_chunks', 0))
        shard_idx = len(manifest['shards'])
        log(f'resuming from manifest: stream cursor {skip_chunks}, '
            f'{manifest.get("total_chunks", 0)} chunks scored, '
            f'{len(manifest["shards"])} shards written', print_console=True)
    else:
        manifest = {'config': vars(args), 'shards': [],
                    'total_tokens': 0, 'total_chunks': 0, 'stream_cursor': 0}
        skip_chunks, shard_idx = 0, 0
    log(f'bulk_score start: {json.dumps(vars(args))}')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    server = None
    server_log = open(HERE / 'bulk_score_server.log', 'wb')
    bar = None
    flush_fn = None
    try:
        log(f'starting server: {args.model} (batch={args.batch})', print_console=True)
        # CREATE_NEW_PROCESS_GROUP: console Ctrl+C must NOT reach the server
        # child, or the graceful drain below has nothing left to talk to.
        server = subprocess.Popen(server_cmd(args), stdout=server_log,
                                  stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        if not wait_healthy():
            raise RuntimeError('server never became healthy; see '
                               'data_pipeline/bulk_score_server.log')
        log('server healthy', print_console=True)

        chunks = chunk_stream(args, tokenizer)
        if skip_chunks:
            t0 = time.time()
            for _ in range(skip_chunks):
                next(chunks)
            log(f'fast-forwarded {skip_chunks} chunks in '
                f'{time.time() - t0:.0f}s', print_console=True)

        # shard accumulators
        tok_acc, ids_acc, probs_acc, mask_acc = [], [], [], []
        c_start, c_len = [], []

        def flush():
            nonlocal shard_idx
            manifest['stream_cursor'] = cursor
            if not tok_acc:
                manifest_path.write_text(json.dumps(manifest, indent=2))
                return
            path = out_dir / f'bulk_{shard_idx:05d}.npz'
            np.savez_compressed(
                path,
                tokens=np.concatenate(tok_acc).astype(np.uint32),
                teacher_ids=np.concatenate(ids_acc),
                teacher_probs=np.concatenate(probs_acc),
                loss_mask=np.concatenate(mask_acc).astype(np.uint8),
                chunk_start=np.array(c_start, dtype=np.int64),
                chunk_length=np.array(c_len, dtype=np.int64),
            )
            ntok = int(sum(c_len))
            manifest['shards'].append({'file': path.name, 'chunks': len(c_len),
                                       'tokens': ntok})
            manifest['total_tokens'] += ntok
            manifest_path.write_text(json.dumps(manifest, indent=2))
            log(f'wrote {path.name}: {len(c_len)} chunks, {ntok} tokens '
                f'(total {manifest["total_tokens"]})', print_console=True)
            shard_idx += 1
            tok_acc.clear(); ids_acc.clear(); probs_acc.clear(); mask_acc.clear()
            c_start.clear(); c_len.clear()

        flush_fn = flush  # arm the exit-path flush BEFORE any failure can occur

        # per-chunk shuffle rng must be per-chunk deterministic even with
        # out-of-order completion: seed it from the chunk ordinal
        def process(ordinal, chunk):
            last_err = None
            for attempt in range(2):  # one retry on transient errors
                try:
                    pp, timings = score_chunk(chunk, args.k)
                    if len(pp) != len(chunk) - 1:
                        raise RuntimeError(
                            f'prompt_probabilities length {len(pp)} != '
                            f'{len(chunk) - 1}')
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(2)
            else:
                raise last_err
            rng = random.Random(args.seed + 1 + ordinal)
            ids, probs = build_rows(pp, len(chunk), args.k, rng)
            mask = np.ones(len(chunk), dtype=np.uint8)
            mask[0] = 0
            return chunk, ids, probs, mask, timings

        n_done = 0
        t_start = time.time()
        scored_tokens = 0
        consec_fail = 0
        cursor = skip_chunks      # next unaccounted-for stream ordinal
        done_ordinals = set()     # completed (scored or dropped) past the cursor
        bar = tqdm(desc='scoring', unit=' tok', unit_scale=True, dynamic_ncols=True)
        with cf.ThreadPoolExecutor(max_workers=args.batch) as pool:
            pending = {}  # future -> ordinal
            ordinal = skip_chunks

            def advance_cursor():
                nonlocal cursor
                while cursor in done_ordinals:
                    done_ordinals.discard(cursor)
                    cursor += 1

            def drain(block=False):
                nonlocal n_done, scored_tokens, consec_fail
                if not pending:
                    return
                wait = cf.FIRST_COMPLETED if block else None
                done, _ = cf.wait(pending, timeout=None if block else 0,
                                  return_when=wait)
                for fut in done:
                    ord_done = pending.pop(fut)
                    done_ordinals.add(ord_done)
                    try:
                        chunk, ids, probs, mask, timings = fut.result()
                    except Exception as e:
                        consec_fail += 1
                        log(f'chunk ordinal {ord_done} FAILED after retry: {e} '
                            f'— dropped ({consec_fail} consecutive)',
                            print_console=True)
                        advance_cursor()
                        if consec_fail >= 3:
                            raise RuntimeError(
                                '3 consecutive chunk failures — server appears '
                                'dead; aborting (progress is flushed)')
                        continue
                    consec_fail = 0
                    c_start.append(sum(c_len))
                    c_len.append(len(chunk))
                    tok_acc.append(np.asarray(chunk, dtype=np.uint32))
                    ids_acc.append(ids)
                    probs_acc.append(probs)
                    mask_acc.append(mask)
                    n_done += 1
                    scored_tokens += len(chunk)
                    manifest['total_chunks'] += 1
                    bar.update(len(chunk))
                    el = time.time() - t_start
                    bar.set_postfix({
                        'tok/s': f'{scored_tokens / max(el, 1e-9):.0f}',
                        'prefill': f'{timings["prompt_per_second"]:.0f}',
                        'chunks': n_done,
                        'shards': shard_idx,
                    })
                    log(f'chunk ordinal {ord_done} done: prefill '
                        f'{timings["prompt_per_second"]:.0f} tok/s')
                    advance_cursor()
                    if n_done % args.chunks_per_npz == 0:
                        flush()

            limit = args.num_chunks
            while not STOP.is_set() and (limit is None or n_done < limit):
                while (not STOP.is_set() and len(pending) < args.batch * 2
                       and (limit is None or ordinal < skip_chunks + limit)):
                    try:
                        chunk = next(chunks)
                    except StopIteration:
                        break
                    pending[pool.submit(process, ordinal, chunk)] = ordinal
                    ordinal += 1
                drain(block=True)

            # graceful stop: finish everything in flight
            while pending:
                drain(block=True)
        flush()
        el = time.time() - t_start
        log(f'done: {n_done} chunks, {scored_tokens} tokens in {el:.0f}s '
            f'({scored_tokens / max(el, 1e-9):.0f} tok/s avg)',
            print_console=True)
    except KeyboardInterrupt:
        # signal handler may not fire under all consoles; this is the fallback
        STOP.set()
        log('KeyboardInterrupt: flushing and shutting down', print_console=True)
    finally:
        if bar is not None:
            bar.close()
        # flush on ANY exit path (success, Ctrl+C, or error) — never lose
        # buffered chunks or leave the manifest cursor stale
        if flush_fn is not None:
            try:
                flush_fn()
            except Exception as e:
                log(f'exit flush failed: {e}', print_console=True)
        if server is not None:
            log('killing server', print_console=True)
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
            subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'],
                           capture_output=True)
        server_log.close()
        log('server down', print_console=True)


def _sigint(signum, frame):
    if STOP.is_set():
        raise KeyboardInterrupt  # second Ctrl+C: hard exit
    log('Ctrl+C received: finishing in-flight chunks, then flushing + exiting '
        '(Ctrl+C again to abort)', print_console=True)
    STOP.set()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _sigint)
    main()
