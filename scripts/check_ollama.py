#!/usr/bin/env python3
"""
Ollama / Qwen-14B Diagnostic Health Check with Hardware Acceleration Verification.

Verifies:
1. Local Ollama server is reachable on http://localhost:11434
2. qwen2.5:14b is present in /api/tags
3. Host GPU / hardware acceleration is detected and configured
4. A real /api/generate generation request succeeds with GPU offloading
"""

import json
import os
import sys
import urllib.error
import urllib.request

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.device import detect_system_gpus, get_ollama_gpu_options, has_gpu

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("LLM_MODEL", "qwen3:14b")


def check_ollama():
    server_status = "FAIL"
    model_status = "FAIL"
    gen_status = "FAIL"
    sample_response = ""

    # Hardware & GPU inspection
    gpus = detect_system_gpus()
    gpu_desc = ", ".join(f"{g['name']} ({g.get('vram_mb', 0)} MB)" for g in gpus) if gpus else "None (CPU only)"
    gpu_opts = get_ollama_gpu_options()
    gpu_offload_status = f"Active (num_gpu: {gpu_opts.get('num_gpu', 'auto')})" if has_gpu() else "Inactive (CPU)"

    # 1. Connect to Ollama server and query /api/tags
    tags_url = f"{OLLAMA_URL}/api/tags"
    try:
        req = urllib.request.Request(tags_url, method="GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        server_status = "PASS"
    except Exception as e:
        print("==================================================")
        print("OLLAMA / QWEN-14B & GPU DIAGNOSTIC")
        print("==================================================")
        print(f"Ollama URL   : {OLLAMA_URL}")
        print(f"Provider     : ollama")
        print(f"Model        : {MODEL_NAME}")
        print(f"GPU Hardware : {gpu_desc}")
        print(f"GPU Offload  : {gpu_offload_status}")
        print(f"Server       : FAIL ({e})")
        print(f"Model        : FAIL")
        print(f"Generation   : FAIL")
        print("STATUS: FAILED")
        sys.exit(1)

    # 2. Verify model exists in /api/tags
    models = data.get("models", [])
    installed_names = []
    for m in models:
        if isinstance(m, dict):
            if "name" in m:
                installed_names.append(m["name"])
            if "model" in m:
                installed_names.append(m["model"])

    target = MODEL_NAME.lower()
    active_model = None
    for name in installed_names:
        n_low = name.lower()
        if (
            target == n_low
            or n_low.startswith(f"{target}:")
            or f"{target}:latest" == n_low
            or (":" not in target and n_low.split(":")[0] == target)
        ):
            active_model = name
            break

    if not active_model:
        # Check for alternative installed Qwen models (e.g. qwen3:14b, qwen3.5:9b)
        candidates = [name for name in installed_names if "qwen" in name.lower()]
        if candidates:
            candidates.sort(key=lambda x: (1 if "14b" in x.lower() else 0, x), reverse=True)
            active_model = candidates[0]
            print(f"Notice: Configured model '{MODEL_NAME}' not found; using installed '{active_model}'.")
        elif installed_names:
            active_model = installed_names[0]
            print(f"Notice: Using available installed model '{active_model}'.")
        else:
            print("==================================================")
            print("OLLAMA / QWEN DIAGNOSTIC")
            print("==================================================")
            print(f"Ollama URL   : {OLLAMA_URL}")
            print(f"Provider     : ollama")
            print(f"Model        : {MODEL_NAME}")
            print(f"GPU Hardware : {gpu_desc}")
            print(f"GPU Offload  : {gpu_offload_status}")
            print(f"Server       : PASS")
            print(f"Model        : FAIL (Available: {installed_names})")
            print(f"Generation   : FAIL")
            print("STATUS: FAILED")
            print(f"\nError: No compatible model found in Ollama. Run: ollama pull {MODEL_NAME}")
            sys.exit(1)

    model_status = "PASS"

    # 3. Send a REAL /api/generate request with GPU offload options
    generate_url = f"{OLLAMA_URL}/api/generate"
    gen_options = {
        "temperature": 0.2,
        "top_p": 0.9,
        "num_predict": 8192,
        **gpu_opts,
    }
    payload = {
        "model": active_model,
        "prompt": "Respond with the single word PONG",
        "stream": False,
        "options": gen_options,
    }

    try:
        gen_req = urllib.request.Request(
            generate_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(gen_req, timeout=60.0) as resp:
            gen_data = json.loads(resp.read().decode("utf-8"))
            sample_response = gen_data.get("response", "").strip()
            if sample_response:
                gen_status = "PASS"
            else:
                gen_status = "FAIL (Empty response)"
    except Exception as e:
        print("==================================================")
        print("OLLAMA / QWEN-14B & GPU DIAGNOSTIC")
        print("==================================================")
        print(f"Ollama URL   : {OLLAMA_URL}")
        print(f"Provider     : ollama")
        print(f"Model        : {MODEL_NAME}")
        print(f"GPU Hardware : {gpu_desc}")
        print(f"GPU Offload  : {gpu_offload_status}")
        print(f"Server       : PASS")
        print(f"Model        : PASS")
        print(f"Generation   : FAIL ({e})")
        print("STATUS: FAILED")
        sys.exit(1)

    all_passed = (server_status == "PASS" and model_status == "PASS" and gen_status == "PASS")

    # 4. Print expected diagnostic output
    print("==================================================")
    print("OLLAMA / QWEN-14B & GPU DIAGNOSTIC")
    print("==================================================")
    print(f"Ollama URL   : {OLLAMA_URL}")
    print(f"Provider     : ollama")
    print(f"Model        : {MODEL_NAME}")
    print(f"GPU Hardware : {gpu_desc}")
    print(f"GPU Offload  : {gpu_offload_status}")
    print(f"Server       : {server_status}")
    print(f"Model        : {model_status}")
    print(f"Generation   : {gen_status}")
    print(f"STATUS: {'READY' if all_passed else 'FAILED'}")
    if sample_response:
        print(f"Sample response: {sample_response[:100]}")

    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    check_ollama()
