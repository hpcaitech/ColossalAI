"""Serve a consolidated diffusers DROID checkpoint to the official RoboLab Cosmos3 client."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import sys
import threading
from http import HTTPStatus
from pathlib import Path

from examples.worldmodel.cosmos3_harness.robolab import checkpoint_fingerprint, load_manifest, verify_checkout

logger = logging.getLogger(__name__)


def serve_policy(policy, metadata: dict, *, host: str, port: int):
    """Return a stoppable WebSocket server; a lock serializes stateful accelerator inference."""
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    from examples.worldmodel.cosmos3_harness.robolab_transport import MAX_MESSAGE_SIZE, pack, unpack

    token = os.environ.get("COSMOS3_API_TOKEN")
    lock = threading.Lock()

    def authenticate(connection, request):
        if token and request.headers.get("Authorization") != f"Bearer {token}":
            return connection.respond(HTTPStatus.UNAUTHORIZED, "Invalid policy token\n")
        return None

    def handle(ws):
        ws.send(pack(metadata))
        for message in ws:
            try:
                observation = unpack(message)
                with lock:
                    response = policy.infer(observation)
                ws.send(pack(response))
            except ConnectionClosed:
                return
            except Exception as exc:
                logger.exception("Policy request failed")
                ws.send(pack({"type": "error", "message": f"{type(exc).__name__}: {exc}"}))

    return serve(
        handle,
        host,
        port,
        process_request=authenticate,
        compression=None,
        max_size=MAX_MESSAGE_SIZE,
        ping_interval=None,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Local diffusers export with transformer/, vae/, text_tokenizer/"
    )
    parser.add_argument("--transformer-path", type=Path, help="Consolidated diffusers transformer export from SFT")
    parser.add_argument("--cosmos-root", type=Path, required=True)
    parser.add_argument("--backend", "--attn-backend", choices=("naive", "cuda", "npu"), required=True)
    parser.add_argument("--device", required=True, help="For example cuda:0 or npu:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    verify_checkout(manifest, args.cosmos_root, "cosmos_framework")
    sys.path.insert(0, str(args.cosmos_root.resolve()))
    # Register the NPU device name before torch.device parses it.
    if args.device.startswith("npu"):
        importlib.import_module("torch_npu")

    from examples.worldmodel.cosmos3_harness.robolab_policy import DroidWamPolicy

    logging.basicConfig(level=logging.INFO)
    logger.info("Hashing checkpoint components for cross-device verification")
    checkpoint_digest = checkpoint_fingerprint(args.checkpoint, args.transformer_path)
    policy = DroidWamPolicy.from_pretrained(
        args.checkpoint.resolve(),
        transformer_path=args.transformer_path,
        config=manifest["policy"],
        device=args.device,
        backend=args.backend,
    )
    metadata = {
        "manifest_fingerprint": manifest["fingerprint"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "checkpoint_fingerprint": checkpoint_digest,
        "backend": args.backend,
        "device": args.device,
        "checkpoint": str(args.checkpoint.resolve()),
        "transformer_path": str(args.transformer_path.resolve()) if args.transformer_path else None,
        "policy": manifest["policy"],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers")},
    }
    with serve_policy(policy, metadata, host=args.host, port=args.port) as server:
        logger.info("Serving RoboLab policy on %s:%s (%s)", args.host, args.port, args.backend)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
