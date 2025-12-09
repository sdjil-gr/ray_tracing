# Multi-GPU Ray Tracing Implementation

## Overview
This document describes the multi-GPU implementation for the CUDA ray tracing program. The modifications enable the program to utilize multiple GPUs simultaneously to accelerate the rendering process.

## Key Changes

### 1. GPU Detection and Initialization
- **Added**: GPU device detection using `cudaGetDeviceCount()`
- **Purpose**: Query the number of available CUDA devices at runtime
- **Behavior**: The program automatically detects and uses all available GPUs

### 2. Image Division Strategy
The image is divided into horizontal strips, with each GPU responsible for rendering one strip:
- Strips are divided as evenly as possible across GPUs
- If the image height is not evenly divisible, the first few GPUs get one extra row
- Each GPU maintains its own strip offset (`y_offset`) for correct coordinate mapping

Example for 1440x1440 image with 3 GPUs:
- GPU 0: rows 0-479 (480 rows)
- GPU 1: rows 480-959 (480 rows)
- GPU 2: rows 960-1439 (480 rows)

### 3. Modified Kernels

#### `init_states` Kernel
**New signature**: `__global__ void init_states(curandStateXORWOW_t* states, int width, int height, int y_offset)`

Changes:
- Added `width`, `height`, and `y_offset` parameters
- Uses `global_index = (y + y_offset) * width + x` to ensure different random seeds for different strips
- This ensures each pixel gets a unique random sequence regardless of which GPU renders it

#### `render` Kernel
**New signature**: `__global__ void render(curandStateXORWOW_t* states, float* image, int width, int height, int samples_per_pixel, int y_offset)`

Changes:
- Added `y_offset` parameter
- Adjusted ray generation: `cam.get_ray(x + r * cos(theta), y + y_offset + r * sin(theta))`
- The `y_offset` ensures rays are cast for the correct pixel coordinates in the global image

### 4. Multi-GPU Execution Flow

1. **Device Initialization Loop**:
   - For each GPU:
     - Copy scene data (camera, spheres, planes, medium) to GPU constant memory
     - Create a CUDA stream for asynchronous execution
     - Allocate random states for the strip
     - Allocate image buffer for the strip
     - Initialize random states kernel

2. **Rendering Loop**:
   - For each sample iteration:
     - For each GPU:
       - Launch render kernel on the GPU's stream with appropriate `y_offset`
     - Synchronize all GPU streams before moving to next iteration
   - Progress reporting shows number of GPUs being used

3. **Result Collection**:
   - Copy each strip from GPU to host memory
   - Combine strips into a single host buffer at correct positions
   - Copy combined image to GPU 0 for final processing
   - Convert float image to uint8 on GPU 0
   - Save final image

### 5. Memory Management

Each GPU maintains:
- Its own `curandStateXORWOW_t*` array for random number generation
- Its own `float*` array for the image strip (float format)
- Its own CUDA stream for asynchronous execution

Host memory:
- Arrays of pointers to manage per-GPU resources
- Temporary buffer for assembling the final image

### 6. Performance Benefits

**Expected Speedup**:
- Near-linear scaling with number of GPUs (e.g., 2x faster with 2 GPUs, 3x with 3 GPUs)
- Actual speedup depends on:
  - GPU specifications (must be similar for balanced workload)
  - PCIe bandwidth for memory transfers
  - Synchronization overhead (minimal as each GPU works independently)

**Overhead**:
- Initial scene data copy to each GPU (negligible, done once)
- Final image assembly on host (minimal compared to rendering time)
- Stream synchronization between sample iterations (necessary for progress reporting)

## Compatibility

### Single GPU Systems
- The code automatically detects and adapts to single GPU systems
- Behaves identically to original implementation when only 1 GPU is available

### Multi-GPU Systems
- Automatically detects and uses all available GPUs
- No configuration required - fully automatic
- Gracefully handles systems with 2, 3, 4, or more GPUs

## Technical Notes

1. **Random Number Generation**: Each pixel receives a globally unique random seed based on its position in the full image, ensuring consistent results regardless of GPU assignment.

2. **Ray Coordinate Mapping**: The `y_offset` parameter is crucial for correct ray generation - it maps local strip coordinates to global image coordinates.

3. **Stream Synchronization**: Synchronization happens after each sample iteration to allow for accurate progress reporting and to maintain deterministic behavior.

4. **Memory Transfer Optimization**: The final image assembly uses efficient `memcpy` to combine strips on the host before transferring the complete image to GPU 0.

## Building and Running

No changes to the build process:
```bash
nvcc -O3 -lcuda -diag-suppress 20012 raytrace.cu -o raytrace -lcudart_static
./raytrace
```

The program will automatically detect and use all available GPUs.

## Output

When running, you'll see:
```
Found N CUDA device(s)
Rendering... X.XX%, remain: M m S s (using N GPU(s))
```

Where N is the number of GPUs being utilized.
