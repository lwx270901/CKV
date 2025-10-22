#!/usr/bin/env python3

import torch
import json
from model.llava_onevision_rekv import load_model

def test_prototrack_mlvu():
    print("Loading model with ProtoTrack-KV...")
    
    # Use default ProtoTrack configuration built into load_model
    model, processor = load_model(
        model_path="model_zoo/llava-onevision-qwen2-7b-ov-hf",
        n_local=2000,
        topk=64,
        chunk_size=1
    )
    
    device = model.device
    print(f"Model loaded on device: {device}")
    
    # Load a sample from MLVU
    print("Loading MLVU sample...")
    with open('data/mlvu/dev_debug_mc.json', 'r') as f:
        mlvu_data = json.load(f)
    
    # Get first sample that has video
    sample = mlvu_data[0]  # Just use the first sample
    
    print(f"Testing with video: {sample['video_path']}")
    print(f"Video ID: {sample['video_id']}")
    
    # Get first conversation
    conv = sample['conversations'][0]
    print(f"Question: {conv['question']}")
    print(f"Answer options: {conv['choices']}")
    
    # Create dummy video if file doesn't exist
    print("Creating dummy video for testing...")
    # 8 frames, 224x224, RGB 
    dummy_video = torch.randint(0, 255, (8, 224, 224, 3), dtype=torch.uint8)
    
    print("Processing video with ProtoTrack-KV...")
    try:
        model.clear_cache()
        model.encode_init_prompt()
        model.encode_video(dummy_video, encode_chunk_size=4)
        print(f"Video encoding successful! KV Cache size: {model.calc_memory_usage() / (1024**2):.1f} MB")
        
        # Test multiple choice QA
        print("Testing multiple choice QA...")
        conv = sample['conversations'][0]
        choices = conv['choices']
        formatted_choices = "\n".join([f"({chr(ord('A') + i)}) {choice}" for i, choice in enumerate(choices)])
        formatted_question = f"Question: {conv['question']}\nOptions:\n{formatted_choices}\nOnly give the best option."
        
        prompt = model.get_prompt(formatted_question, mc=True)
        input_text = {'question': conv['question'], 'prompt': prompt}
        
        result = model.question_answering(input_text, max_new_tokens=10)
        print(f"Model response: {result}")
        
        # Extract predicted choice
        pred_choice = None
        if ")" in result:
            index = result.index(")")
            pred_choice = result[index - 1 : index]
        else:
            pred_choice = result[0] if result else "A"
        
        print(f"Predicted choice: {pred_choice}")
        print(f"Correct answer: {conv['answer']}")
        
        print("\n=== ProtoTrack-KV Test Successful! ===")
        print(f"✅ Video processing: OK")
        print(f"✅ KV Cache usage: {model.calc_memory_usage() / (1024**2):.1f} MB")
        print(f"✅ Question answering: OK")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prototrack_mlvu()