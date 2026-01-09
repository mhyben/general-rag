# Troubleshooting Guide

## PyTorch CUDA Timing Error

If you encounter the error:
```
terminate called after throwing an instance of 'c10::Error'
what(): fast_1 >= fast_0 INTERNAL ASSERT FAILED
```

This is a known PyTorch/CUDA timing issue. Here are several solutions:

### Solution 1: Use CPU-Only Mode (Recommended)

Force PyTorch to use CPU instead of CUDA:

```bash
export CUDA_VISIBLE_DEVICES=""
python rag_cli.py
```

Or set it in your shell profile:
```bash
echo 'export CUDA_VISIBLE_DEVICES=""' >> ~/.bashrc
source ~/.bashrc
```

### Solution 2: Reinstall PyTorch

The issue may be caused by PyTorch version conflicts. Try reinstalling:

```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or if you need CUDA, reinstall with specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Solution 3: Update PyTorch

Update to the latest PyTorch version:

```bash
pip install --upgrade torch torchvision torchaudio
```

### Solution 4: Set Environment Variables

Before running the script:

```bash
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=0
python rag_cli.py
```

### Solution 5: Use Conda Environment

If using conda, try creating a fresh environment:

```bash
conda create -n rag-env python=3.10
conda activate rag-env
pip install -r requirements.txt
```

### Solution 6: Check for Multiple PyTorch Installations

Check if you have multiple PyTorch installations:

```bash
pip list | grep torch
conda list | grep torch
```

Remove duplicates and reinstall.

## Other Common Issues

### Import Errors

If you get import errors, make sure you're in the correct directory:

```bash
cd RAG
python rag_cli.py
```

### Ollama Connection Issues

If Ollama connection fails:

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Check if Ollama is accessible:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Memory Issues

If you run out of memory:

1. Use a smaller embedding model (e.g., `all-MiniLM-L6-v2`)
2. Process fewer documents at once
3. Use CPU mode instead of GPU

## Getting Help

If none of these solutions work, please:
1. Check your PyTorch version: `python -c "import torch; print(torch.__version__)"`
2. Check your CUDA version: `nvidia-smi` (if using GPU)
3. Check Python version: `python --version`
4. Share the full error message and your environment details

