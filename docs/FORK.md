# Bortlesboat ReActor fork

This fork builds from Gourieff/ComfyUI-ReActor's Python source. The changes are on the `improve/reliability` branch.

## NSFW detector switch

Both **ReActor Fast Face Swap** and **ReActor Fast Face Swap [OPTIONS]** expose `nsfw_detection` with ON/OFF labels. It defaults to **ON**, including when an older workflow omits the new input. Use **OFF** to skip the detector.

When OFF, a swap does not load the detector, download its assets, import Transformers for detection, or classify input images. It also releases any classifier retained from an earlier checked swap. The detector's weight file is approximately **328 MiB**; fresh installations that keep detection off avoid that download. Existing shared model files are retained.

When ON, the classifier is loaded once for the active model path and device. Repeated images reuse it. Changing the device or model path reloads it, and **ReActor Unload Models** releases it. The upstream `0.979` threshold is retained. A rejected image reports an error identifying its position in the batch; the batch is not silently shortened or replaced by a black image.

## Setup

Use one ReActor checkout per ComfyUI installation. Back up an existing checkout before replacing it. From the `custom_nodes` directory:

```sh
git clone --branch improve/reliability https://github.com/Bortlesboat/ComfyUI-ReActor.git ComfyUI-ReActor
```

Follow the upstream installation steps below the fork notice in the main README, using ComfyUI's Python environment. The existing installer can change runtime dependencies, so preserve a working environment before running it. Restart ComfyUI and refresh its browser page to load the switch.

The source archive excludes downloaded model weights and third-party DLLs. ONNX face-swap models and restoration assets are still needed for their corresponding features. Original code and model license terms remain applicable; this fork is not registered under the upstream publisher's identity.

## Reliability changes

Model downloads use one HTTP request and a temporary file in the destination directory. Empty or announced-size-incomplete transfers raise an error. Successful downloads atomically replace the destination; failures remove temporary files and preserve an existing model. This shared downloader is used by both the extension and installer. Transfers use a 60-second socket timeout.

Disabled nodes and missing-source bypasses return all three declared outputs, including `ORIGINAL_IMAGE`. User interruption raises ComfyUI's cancellation exception.

## Checks and source rebuild

The lightweight tests require only Pillow and tqdm:

```sh
python -m pip install pillow tqdm
python tests/run.py
python -m compileall -q .
```

Node integration tests additionally require the normal ComfyUI/ReActor dependencies. From this checkout, set `COMFYUI_ROOT` to the directory containing ComfyUI's `folder_paths.py`, then use its Python environment:

```powershell
$env:COMFYUI_ROOT = (Resolve-Path '../..').Path
python tests/run.py
```

The runtime tests use temporary model directories and controlled swap/classifier backends. They verify the node contracts and switch independently of downloaded model weights. Lightweight CI explicitly skips those integration tests when `COMFYUI_ROOT` is absent.

Create a source bundle from a committed revision:

```sh
git archive --format=zip --prefix=ComfyUI-ReActor/ -o ../ComfyUI-ReActor-source.zip HEAD
```

Use a saved commit ID to rebuild that exact version. No decompiler is needed for the published Python source; separately supplied third-party native libraries are outside this source rebuild.
