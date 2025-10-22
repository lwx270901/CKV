#!/usr/bin/env python3
"""
Minimal test script to verify ProtoTrack-KV integration
"""
import torch
from model.llava_onevision_rekv import load_model

def test_prototrack_basic():
    print("Loading model with ProtoTrack-KV...")
    model, processor = load_model(
        model_path='model_zoo/llava-onevision-qwen2-7b-ov-hf', 
        n_local=2000
    )
    
    print("Clearing cache...")
    model.clear_cache()
    
    print("Encoding init prompt...")
    model.encode_init_prompt()
    
    print("Creating dummy video tensor...")
    # Create a small dummy video: 4 frames, 224x224, RGB
    dummy_video = torch.randint(0, 255, (4, 224, 224, 3), dtype=torch.uint8)
    
    print("Encoding video...")
    try:
        model.encode_video(dummy_video, encode_chunk_size=2)
        print(f"Video encoding successful! KV Cache size: {model.calc_memory_usage() / (1024**2):.1f} MB")
        
        # Test QA
        print("Testing question answering...")
        question = "What do you see in the video?"
        prompt = model.get_prompt(question)
        print(f"Question: {question}")
        print("Running QA...")
        
        result = model.question_answering({'question': question, 'prompt': prompt}, max_new_tokens=5)
        print(f"QA result: {result}")
        
    except Exception as e:
        print(f"Error during video encoding: {e}")
        import traceback
        traceback.print_exc()
    
    print("Test complete!")

if __name__ == "__main__":
    test_prototrack_basic()