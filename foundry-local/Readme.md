# Microsoft Foundry Local – Installation and Setup Guide

## Overview

Microsoft Foundry Local enables developers to run AI models locally on their machines without depending on cloud-hosted inference endpoints.

However, getting it up and running involves more than simply installing the CLI. Model variants, hardware compatibility, execution providers, model loading, and local inference configuration all play an important role.

This guide documents the installation process, essential CLI commands, model discovery, and validation steps for running local LLMs.

## 1. Install Microsoft Foundry Local

Open PowerShell or Windows Terminal and execute:

```powershell
winget install Microsoft.FoundryLocal
```

Once installation completes, verify the installation:

```powershell
foundry
```

This should display the Foundry Local CLI help and available commands.

## 2. Manage the Foundry Local Server

Foundry Local runs a local daemon that manages model discovery, model loading, and inference requests.

Start the server:

```powershell
foundry server start
```

Stop the server:

```powershell
foundry server stop
```

Check the server status:

```powershell
foundry server status
```

Check the overall Foundry Local status:

```powershell
foundry status
```

## 3. Explore Available Models

List the available models:

```powershell
foundry model list
```

Search for a specific model:

```powershell
foundry model list --search qwen2.5-0.5b
```

### Important: Identify CPU and GPU variants

This is one of the most important steps when working with Foundry Local.

A model can have multiple execution variants depending on the available hardware and execution provider.

Use the `--variants` option to discover them:

```powershell
foundry model list --search <model-name> --variants
```

Examples:

```powershell
foundry model list --search qwen2.5-0.5b --variants

foundry model list --search phi-3.5-mini --variants
```

**Hardware compatibility considerations:**

- GPU variants depend on supported hardware, drivers, and execution providers.
- Generic CPU variants are useful when GPU execution is unavailable or incompatible.
- For my laptop-based setup, the generic CPU variants were the working option. The OpenVINO GPU variants encountered an ONNX Runtime execution-provider compatibility issue.

Do not assume that every GPU variant will work simply because the model appears in the catalog.

## 4. Run Models Locally

Once you identify a suitable model variant, you can run it directly.

Examples:

```powershell
foundry run Phi-3.5-mini-instruct-generic-cpu

foundry run qwen2.5-0.5b-instruct-generic-cpu
```

## 5. Load the Model Explicitly

**Important:** If you intend to consume the model through the local REST API or an OpenAI-compatible client, explicitly load the model first.

Example:

```powershell
foundry model load qwen2.5-0.5b-instruct-generic-cpu
```

This ensures that the model is loaded into the Foundry Local inference runtime before sending API requests.

## 6. Validate Local Inference

You can validate inference directly from the CLI using the `complete` command:

```powershell
foundry complete qwen2.5-0.5b-instruct-generic-cpu "Explain Azure in three sentences"
```

If the model returns a response, your local inference setup is operational.

## 7. Useful Configuration and Cache Commands

Inspect the Foundry Local configuration:

```powershell
foundry config show
```

Find the model cache location:

```powershell
foundry cache location
```

List available models:

```powershell
foundry model list
```

## 8. Quick Setup Checklist

- [ ] Install Microsoft Foundry Local.
- [ ] Verify the CLI installation.
- [ ] Start the Foundry Local server.
- [ ] Verify server status.
- [ ] Discover available model variants.
- [ ] Select a hardware-compatible model variant.
- [ ] Run the model locally.
- [ ] Explicitly load the model for API-based inference.
- [ ] Validate inference using the CLI.

## Conclusion

Foundry Local provides a convenient way to experiment with local LLM inference, but hardware compatibility and model execution-provider selection are critical.

My recommendation is to **always inspect model variants before loading a model**, particularly when working with laptops or development workstations with integrated GPUs.

This guide is based on hands-on experimentation with Foundry Local on Windows, including CPU-based inference using Qwen 2.5 0.5B.

Happy experimenting with local AI! 🚀
