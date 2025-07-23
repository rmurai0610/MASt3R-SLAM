#!/usr/bin/env python3
import sys
import torch
sys.path.append('/app')
sys.path.append('/app/thirdparty/mast3r')

from mast3r.model import AsymmetricMASt3R

# Try to load the model with verbose error handling
try:
    weights_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    print(f"Loading model from: {weights_path}")
    
    # Load the checkpoint first to inspect it
    checkpoint = torch.load(weights_path, map_location='cpu')
    print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}...")  # Show first 5 keys
    
    if 'model' in checkpoint:
        print("Found 'model' key in checkpoint")
        if 'config' in checkpoint:
            print(f"Model config: {checkpoint['config']}")
    
    # Try to load with the actual method
    model = AsymmetricMASt3R.from_pretrained(weights_path)
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()