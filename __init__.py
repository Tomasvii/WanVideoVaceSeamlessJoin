from .combine_video_clips_extended import CombineVideoClips
from .seamless_join_video_clips import WanVideoVaceSeamlessJoin

NODE_CLASS_MAPPINGS = {
    "CombineVideoClips": CombineVideoClips,
    "WanVideoVaceSeamlessJoin": WanVideoVaceSeamlessJoin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineVideoClips": "Combine Video Clips (Extended)",
    "WanVideoVaceSeamlessJoin": "WanVideo Vace Seamless Join",
}
