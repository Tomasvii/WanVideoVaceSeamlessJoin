# WanVideoVaceSeamlessJoin/__init__.py

from .combine_video_clips_extended import CombineVideoClipsExtended
from .seamless_join_video_clips import SeamlessJoinVideoClips

NODE_CLASS_MAPPINGS = {
    "CombineVideoClipsExtended": CombineVideoClipsExtended,
    "SeamlessJoinVideoClips": SeamlessJoinVideoClips,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineVideoClipsExtended": "Combine Video Clips (Extended)",
    "SeamlessJoinVideoClips": "Seamless Join Video Clips",
}
