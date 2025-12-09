# Multi-GPU Ray Tracing - Visual Architecture

```
Original Single-GPU Architecture:
==================================

┌─────────────────────────────────────┐
│         GPU 0                       │
│  ┌───────────────────────────────┐  │
│  │   Full Image (1440x1440)      │  │
│  │   Random States               │  │
│  │   Render Kernel               │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
         │
         ▼
    Final Image


New Multi-GPU Architecture (Example: 3 GPUs):
==============================================

┌─────────────────────────────────────┐
│         GPU 0                       │
│  ┌───────────────────────────────┐  │
│  │   Strip 0: rows 0-479         │  │
│  │   (480 rows)                  │  │
│  │   y_offset = 0                │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         GPU 1                       │
│  ┌───────────────────────────────┐  │
│  │   Strip 1: rows 480-959       │  │
│  │   (480 rows)                  │  │
│  │   y_offset = 480              │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         GPU 2                       │
│  ┌───────────────────────────────┐  │
│  │   Strip 2: rows 960-1439      │  │
│  │   (480 rows)                  │  │
│  │   y_offset = 960              │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
         │
         ▼
    ┌───────────────┐
    │ Host Assembly │
    │  (CPU Memory) │
    └───────────────┘
         │
         ▼
    Final Image


Execution Flow:
===============

1. Initialization Phase:
   ┌──────────────┐
   │ Detect GPUs  │
   └──────┬───────┘
          │
          ▼
   ┌──────────────────────────┐
   │ Divide Image into Strips │
   └──────┬───────────────────┘
          │
          ▼
   ┌────────────────────────────┐
   │ For Each GPU:              │
   │  - Set Device              │
   │  - Copy Scene Data         │
   │  - Create Stream           │
   │  - Allocate Buffers        │
   │  - Init Random States      │
   └────────────────────────────┘

2. Rendering Phase (Repeated for each sample):
   ┌───────────────────────────┐
   │ For Each GPU:             │
   │  - Launch Render Kernel   │
   │    (with y_offset)        │
   └────────┬──────────────────┘
            │
            ▼
   ┌───────────────────────────┐
   │ Synchronize All GPUs      │
   └────────┬──────────────────┘
            │
            ▼
   ┌───────────────────────────┐
   │ Update Progress Display   │
   └───────────────────────────┘

3. Assembly Phase:
   ┌────────────────────────────┐
   │ For Each GPU:              │
   │  - Copy Strip to Host      │
   │  - Place in Full Image     │
   └────────┬───────────────────┘
            │
            ▼
   ┌────────────────────────────┐
   │ Copy Combined Image to GPU │
   └────────┬───────────────────┘
            │
            ▼
   ┌────────────────────────────┐
   │ Convert Float to Uint8     │
   └────────┬───────────────────┘
            │
            ▼
   ┌────────────────────────────┐
   │ Save Image to File         │
   └────────────────────────────┘


Ray Coordinate Mapping:
========================

Local Strip Coordinate (x, y):
- x: 0 to width-1 (same for all strips)
- y: 0 to strip_height-1 (local to each strip)

Global Image Coordinate:
- x: same as local x
- y: local_y + y_offset

Example for pixel at strip position (100, 50) on GPU 1:
- Local coordinates: (100, 50)
- y_offset: 480
- Global coordinates: (100, 530)
- Ray generation: cam.get_ray(100, 530)


Memory Layout:
==============

GPU 0 Memory:
┌────────────────────────────────────┐
│ Constant Memory:                   │
│  - camera                          │
│  - spheres[MAX_ITEM_COUNT]         │
│  - planes[MAX_ITEM_COUNT]          │
│  - medium                          │
├────────────────────────────────────┤
│ Device Memory:                     │
│  - states[width * strip_h[0]]      │
│  - image_f[width * strip_h[0] * 3] │
└────────────────────────────────────┘

GPU 1 Memory:
┌────────────────────────────────────┐
│ Constant Memory:                   │
│  - camera                          │
│  - spheres[MAX_ITEM_COUNT]         │
│  - planes[MAX_ITEM_COUNT]          │
│  - medium                          │
├────────────────────────────────────┤
│ Device Memory:                     │
│  - states[width * strip_h[1]]      │
│  - image_f[width * strip_h[1] * 3] │
└────────────────────────────────────┘

... (repeated for each GPU)

Host Memory:
┌────────────────────────────────────┐
│ image_f_host[width * height * 3]   │
│                                    │
│  ┌──────────────────────────────┐  │
│  │ Strip 0 (from GPU 0)         │  │
│  ├──────────────────────────────┤  │
│  │ Strip 1 (from GPU 1)         │  │
│  ├──────────────────────────────┤  │
│  │ Strip 2 (from GPU 2)         │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘


Performance Model:
==================

Single GPU Time: T
Number of GPUs: N

Ideal Speedup: T / N

Actual Time ≈ T/N + overhead
where overhead = 
  - Scene data copy to each GPU (negligible)
  - Final image assembly (< 1% of T)
  - Stream synchronization per iteration (minimal)

Expected Speedup: ~N (near-linear)

Example with 1440x1440 image, 16384 samples:
- 1 GPU:  ~T seconds
- 2 GPUs: ~T/2 seconds (1.9x-2.0x speedup)
- 3 GPUs: ~T/3 seconds (2.8x-3.0x speedup)
- 4 GPUs: ~T/4 seconds (3.7x-4.0x speedup)
```
