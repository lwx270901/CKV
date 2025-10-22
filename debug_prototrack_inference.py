#!/usr/bin/env python3
"""
Debug ProtoTrack-KV inference
Simple debug script to test ProtoTrack-KV inference directly
"""

import sys
import os
import json
sys.path.append('/home/minh/research/ReKV')

def test_prototrack_inference():
    """Test ProtoTrack-KV inference directly"""
    
    print("Testing ProtoTrack-KV inference...")
    
    # Load model
    try:
        from model.llava_onevision_rekv import load_model
        print("Loading ProtoTrack-KV model...")
        
        model, processor = load_model(
            model_path='model_zoo/llava-onevision-qwen2-7b-ov-hf',
            n_local=8000,
            topk=32
        )
        print("✓ Model loaded successfully")
        
        # Load a single test question
        data_path = 'data/mlvu/dev_debug_mc.json'
        with open(data_path, 'r') as f:
            questions = json.load(f)
        
        if not questions:
            print("No questions found")
            return
            
        test_question = questions[0]
        print(f"Test question: {test_question}")
        
        # Test inference with simple approach
        try:
            # Try direct inference
            if hasattr(model, 'answer'):
                answer = model.answer(test_question['question'])
                print(f"Direct answer: {answer}")
                print(f"Answer type: {type(answer)}")
            else:
                print("Model doesn't have answer method")
                
            # Try with video if available
            video_path = test_question.get('video')
            if video_path:
                video_full_path = f"data/mlvu/videos/{video_path}"
                if os.path.exists(video_full_path):
                    print(f"Video found: {video_full_path}")
                    # Test video inference
                    if hasattr(model, 'ingest_video'):
                        model.ingest_video(video_full_path)
                        answer = model.answer(test_question['question'])
                        print(f"Video answer: {answer}")
                        print(f"Video answer type: {type(answer)}")
                else:
                    print(f"Video not found: {video_full_path}")
        
        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Model loading error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prototrack_inference()