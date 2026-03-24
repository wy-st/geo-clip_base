# Lightweight public API — only expose the inference loader.
# Training / Lightning / torchgeo dependencies are NOT imported here.
from .load_satclip import load_satclip_loc_encoder  # noqa: F401
