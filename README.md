# Custom Nodes That Power This Magic

# Installation
```
git clone https://github.com/Tomasvii/WanVideoVaceSeamlessJoin.git .
```

Clone the two .py files directly in custom_nodes, no subfolder needed.

# 1. WanVideoVaceSeamlessJoin Node

The secret sauce for creating long-form videos without transitions breaks.

What it does:

    Intelligently analyzes overlapping frames between video clips

    Creates smooth transitions using advanced masking

    Eliminates the "cut" feeling between generated segments

    Maintains motion continuity across joins

Key Features:

    mask_first_frames: Controls transition smoothness

    mask_last_frames: Prevents jarring cuts

    frame_load_cap: Optimizes processing for any video length

    Automatic path handling for batch processing

# 2. CombineVideoClips Node

Your one-stop solution for merging multiple video segments into cinematic sequences.

What it does:

    Combines up to 6 video files into a single seamless sequence

    Intelligent frame management and optimization

    Preserves quality while merging different clips

    Perfect for creating extended narratives

Key Features:

    Multiple video input support (first, second, third, etc.)

    Automatic frame rate synchronization

    Quality preservation during combination

    Batch processing capabilities
