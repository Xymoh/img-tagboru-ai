# GPU Setup Guide for Ollama

## Why GPU?
- **Without GPU (CPU only)**: 14B models take **5-30 minutes per generation** ❌ VERY SLOW
- **With GPU (NVIDIA/AMD)**: Same models take **10-60 seconds** ✅ FAST

## GPU Setup by Platform

### Windows + NVIDIA GPU

1. **Check your GPU**:
   - Open `Device Manager` → `Display adapters`
   - Look for NVIDIA GeForce/Tesla/Quadro card

2. **Install NVIDIA CUDA Toolkit**:
   - Download: https://developer.nvidia.com/cuda-downloads
   - Select: Windows, x86_64, 11.8 or latest
   - Install with default settings

3. **Verify Installation**:
   ```bash
   nvidia-smi
   ```
   - You should see your GPU info and CUDA version

4. **Start Ollama with GPU**:
   - Ollama automatically uses GPU when installed correctly
   - Run: `ollama serve`
   - Check: `ollama ps` → should show "GPU loaded"

### Windows + AMD GPU

1. **Install AMD Radeon GPU drivers**

2. **Install ROCm for AMD** (optional, for better performance):
   - Download: https://rocmdocs.amd.com/en/latest/deploy/windows/quick_start.html
   
3. **Start Ollama**: It will auto-detect AMD GPU

### No GPU / CPU Only

For CPU-only systems, use smaller models:
- `qwen2:1.5b` - Fast (10-30 seconds)
- `qwen2:7b` - Medium (2-10 minutes)  
- `qwen2:14b` - Slow (10-60 minutes) ← Not recommended for CPU

## Troubleshooting

**GPU not detected:**
1. Check `nvidia-smi` shows your GPU
2. Restart Ollama: `ollama serve`
3. In browser, visit: http://localhost:11434/api/tags
4. Verify VRAM allocated (check Windows Task Manager → GPU tab)

**Out of VRAM:**
- Stop other GPU apps (games, miners, etc.)
- Use smaller model (7B instead of 14B)
- Reduce batch size in Ollama config

**Performance still slow:**
- Switch to smaller model
- Increase RAM (larger models need more system RAM too)
- Check Task Manager → GPU tab (should be >50% utilized)
