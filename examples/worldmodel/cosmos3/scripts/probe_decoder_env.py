"""Which video decoders the official-stack dependencies would resolve to on this host.

The upstream DROID dataset decodes through ``lerobot.datasets.video_utils.decode_video_frames``, so what that
function does on this machine is what "use the official pipeline" would actually mean here. Written as a file
because ``dev-container-node.sh`` forwards its arguments by joining them into one ``bash -c`` string, which
destroys inline multi-line ``python -c`` code.
"""

import importlib.util


def status(module):
    spec = importlib.util.find_spec(module)
    return "present" if spec else "absent"


for name in ("lerobot", "torchcodec", "decord", "av", "torchvision", "torch", "torch_npu"):
    line = f"{name:12s} {status(name)}"
    if status(name) == "present":
        module = __import__(name)
        line += f"  version={getattr(module, '__version__', 'unknown')}"
    print(line)

if status("lerobot") == "present":
    from lerobot.datasets import video_utils

    print("\ndecode_video_frames source file:", video_utils.__file__)
    try:
        import torchcodec.decoders  # noqa: F401

        print("lerobot would decode through torchcodec")
    except Exception as error:  # noqa: BLE001
        print(f"lerobot cannot use torchcodec: {type(error).__name__}: {error}")
