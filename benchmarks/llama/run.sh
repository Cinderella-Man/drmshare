#!/bin/bash
# llama.cpp (Vulkan) on two GPUs: tensor-parallel with the all-reduce through host RAM vs through shared video memory
# (GGML_VK_ALLREDUCE_P2P=1, needs the DRMShare nvidia-drm loaded), plus layer split as a reference. Same binary for
# every setting. Writes perplexity logs, llama-bench tables, one llama-server chat answer and the kernel log count per
# step into OUT.
#
# usage: MODEL=<gguf> TEXT=<text file> [LLAMA_BIN=<dir>] [OUT=<dir>] [TS=a,b] [ONLY=all|bench|chat] ./run.sh
#   LLAMA_BIN  llama-bench, llama-perplexity, llama-server built from tools/llama.cpp (default: its build-vk/bin)
#   TS         tensor split, Vulkan device order (default: by each card's memory, e.g. 6144 + 12288 MiB -> 6,12)
#   Run as root to count the driver's kernel log lines per step (dmesg); otherwise that column is skipped.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
B=${LLAMA_BIN:-$HERE/../../tools/llama.cpp/build-vk/bin}
M=${MODEL:?set MODEL to a .gguf file}
TXT=${TEXT:?set TEXT to a text file for the perplexity check}
O=${OUT:-$HERE/results/local/$(date +%Y-%m-%d)-$(basename "$M" .gguf)}  # local/: ignored by git (raw logs hold paths)
ONLY=${ONLY:-all}   # bench: only llama-bench; chat: only the llama-server chat
mkdir -p "$O" && cd "$O" || exit 1
unset GGML_VK_ALLREDUCE_P2P
ROOT=0; [ "$(id -u)" = 0 ] && ROOT=1

echo "== nvidia_drm peer_vidmem_dmabuf = $(cat /sys/module/nvidia_drm/parameters/peer_vidmem_dmabuf 2>/dev/null || echo 'not loaded (shared-memory runs will fall back)')"
"$B"/llama-bench --list-devices 2>&1 | grep -E "Vulkan[0-9]" | tee devices.txt
if [ -z "${TS:-}" ]; then
  TS=$(grep -o -E '\(([0-9]+) MiB' devices.txt | grep -o -E '[0-9]+' | awk '{printf "%s%d", (NR>1?",":""), $1/1024}')
fi
echo "== tensor split $TS (Vulkan device order)"
nvrm() { [ $ROOT = 1 ] || return 0; echo "     kernel log during this step: $(dmesg | grep -c NVRM) NVRM lines"; dmesg -C; }
[ $ROOT = 1 ] && dmesg -C

ppl() {  # $1 = label, rest = environment
  local L=$1; shift
  env "$@" "$B"/llama-perplexity -m "$M" -f "$TXT" -ngl 99 -sm tensor -ts "$TS" -c 512 --chunks 8 -v > ppl-$L.log 2>&1
  echo "-- perplexity $L: exit $? | $(grep -h -E 'Final estimate' ppl-$L.log | cut -c16-80) | $(grep -h -o -E 'all-reduce writes straight[^|]*|P2P all-reduce: [0-9]+ MiB[^|]*' ppl-$L.log | sort -u | tr '\n' ' ')"
  nvrm
}
bench() {  # $1 = label, $2 = split mode, rest = environment
  local L=$1 S=$2; shift 2
  # llama-bench separates one split's values with '/' (a ',' would mean two separate configurations)
  env "$@" "$B"/llama-bench -m "$M" -ngl 99 -sm "$S" -ts "${TS//,//}" -p 512,2048 -n 128 -r 3 -o md > bench-$L.md 2>&1
  echo "-- llama-bench $L: exit $?"; grep -E '^\|' bench-$L.md | sed 's/^/     /'
  nvrm
}

if [ "$ONLY" = all ]; then
  ppl tensor-hostram
  ppl tensor-shared GGML_VK_ALLREDUCE_P2P=1
fi
if [ "$ONLY" != chat ]; then
  for round in 1 2; do
    bench tensor-hostram-r$round tensor
    bench tensor-shared-r$round tensor GGML_VK_ALLREDUCE_P2P=1
  done
  bench layer layer
fi
if [ "$ONLY" != bench ]; then
  # A chat answer through llama-server, tensor-parallel with the shared-memory all-reduce. The first request on a new
  # driver compiles Vulkan pipelines and is slow; the second one is the one to compare.
  PORT=${PORT:-8099}
  GGML_VK_ALLREDUCE_P2P=1 "$B"/llama-server -v -m "$M" -ngl 99 -sm tensor -ts "$TS" -c 4096 --host 127.0.0.1 \
    --port $PORT > server.log 2>&1 & SV=$!
  for i in $(seq 1 180); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200 && break; sleep 1; done
  for req in 1 2; do
    curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "messages": [{"role": "user", "content": "In three sentences: why does splitting a language model across two GPUs need the GPUs to exchange data after every layer?"}],
      "max_tokens": 300, "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": false}}' > chat$req.raw
    # keep the answer, the timings and the model's file name (the full reply carries the model's path)
    python3 -c 'import json,os,sys; r=json.load(open(sys.argv[1])); t=r.get("timings",{}); json.dump({"model": os.path.basename(r.get("model","")), "answer": r["choices"][0]["message"]["content"], "timings": t}, open(sys.argv[2],"w"), indent=1); print("-- chat request %s: prompt %d tok %.1f t/s, generated %d tok %.1f t/s" % (sys.argv[3], t.get("prompt_n",0), t.get("prompt_per_second",0), t.get("predicted_n",0), t.get("predicted_per_second",0)))' chat$req.raw chat$req.json $req \
      && rm -f chat$req.raw || { echo "-- chat failed"; tail -5 server.log; }
  done
  kill $SV; wait $SV 2>/dev/null
  echo "-- server log: $(grep -h -o -E 'all-reduce writes straight[^|]*|P2P all-reduce: [0-9]+ MiB[^|]*' server.log | sort -u | tr '\n' ' ')"
  nvrm
fi
echo "== done: $O"
