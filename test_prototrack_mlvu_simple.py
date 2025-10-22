#!/usr/bin/env python3

import os
import json
import torch
import pandas as pd
from pathlib import Path

# Add the project root to path  
import sys
sys.path.append('/home/minh/research/ReKV')

from model.llava_onevision_rekv import load_model

def test_prototrack_mlvu_simple():
    """Simple test of ProtoTrack-KV on MLVU dataset"""
    
    print("=== ProtoTrack-KV MLVU Simple Test ===")
    
    # Load model
    print("Loading ProtoTrack-KV model...")
    model, processor = load_model(
        model_path="model_zoo/llava-onevision-qwen2-7b-ov-hf",
        n_local=2000,
        topk=64,
        chunk_size=1
    )
    print(f"✅ Model loaded on {model.device}")
    
    # Load MLVU data
    print("Loading MLVU dataset...")
    with open('data/mlvu/dev_debug_mc.json', 'r') as f:
        mlvu_data = json.load(f)
    
    # Test with first 3 samples
    test_samples = mlvu_data[:3]
    print(f"Testing with {len(test_samples)} samples")
    
    results = []
    
    for i, sample in enumerate(test_samples):
        print(f"\n--- Sample {i+1}: {sample['video_id']} ---")
        
        try:
            # Create dummy video
            dummy_video = torch.randint(0, 255, (6, 224, 224, 3), dtype=torch.uint8)
            print(f"Created dummy video: {dummy_video.shape}")
            
            # Encode video with ProtoTrack
            print("Encoding with ProtoTrack-KV...")
            model.clear_cache()
            model.encode_init_prompt()
            model.encode_video(dummy_video, encode_chunk_size=3)
            
            cache_size = model.calc_memory_usage() / (1024**2)
            print(f"✅ Video encoded. Cache size: {cache_size:.1f} MB")
            
            # Test first question
            conv = sample['conversations'][0]
            question = conv['question']
            choices = conv['choices']
            correct_answer = conv['answer']
            
            print(f"Question: {question}")
            print(f"Choices: {choices}")
            print(f"Correct: {correct_answer}")
            
            # Format multiple choice question
            choice_letters = ['A', 'B', 'C', 'D']
            formatted_choices = "\n".join([f"({choice_letters[j]}) {choice}" for j, choice in enumerate(choices)])
            formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nAnswer with just the letter."
            
            # Get model response
            prompt = model.get_prompt(formatted_question, mc=True)
            input_text = {'question': question, 'prompt': prompt}
            
            print("Generating answer...")
            pred_answer = model.question_answering(input_text, max_new_tokens=5)
            print(f"Model response: '{pred_answer}'")
            
            # Extract predicted choice
            pred_choice = 'A'  # default
            pred_clean = pred_answer.strip().upper()
            if pred_clean and pred_clean[0] in choice_letters:
                pred_choice = pred_clean[0]
            elif ")" in pred_answer:
                idx = pred_answer.index(")")
                if idx > 0:
                    pred_choice = pred_answer[idx-1].upper()
            
            # Check if correct
            if correct_answer in choices:
                correct_choice = choice_letters[choices.index(correct_answer)]
            else:
                correct_choice = 'A'
            
            is_correct = pred_choice == correct_choice
            print(f"Predicted: {pred_choice}, Correct: {correct_choice}, Match: {is_correct}")
            
            # Store result
            results.append({
                'video_id': sample['video_id'],
                'question': question,
                'correct_answer': correct_answer,
                'pred_choice': pred_choice,
                'correct_choice': correct_choice,
                'is_correct': is_correct,
                'cache_size_mb': cache_size,
                'pred_answer_raw': pred_answer
            })
            
            print("✅ Sample completed successfully")
            
        except Exception as e:
            print(f"❌ Error processing sample: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n=== Results Summary ===")
    if results:
        df = pd.DataFrame(results) 
        
        # Save results
        os.makedirs('results/prototrack_test', exist_ok=True)
        df.to_csv('results/prototrack_test/mlvu_simple_test.csv', index=False)
        
        correct_count = sum(r['is_correct'] for r in results)
        accuracy = correct_count / len(results) * 100
        avg_cache_size = df['cache_size_mb'].mean()
        
        print(f"Samples processed: {len(results)}")
        print(f"Correct answers: {correct_count}/{len(results)}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Average cache size: {avg_cache_size:.1f} MB")
        
        print(f"\nDetailed results:")
        for r in results:
            status = "✅" if r['is_correct'] else "❌"
            print(f"{status} {r['video_id']}: {r['pred_choice']} vs {r['correct_choice']} ({r['cache_size_mb']:.1f}MB)")
        
        print(f"\nResults saved to: results/prototrack_test/mlvu_simple_test.csv")
    
    print("\n=== ProtoTrack-KV MLVU Test Complete ===")

if __name__ == "__main__":
    test_prototrack_mlvu_simple()