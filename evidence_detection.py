"""
Advanced Evidence Detection for QAR Measurement
Implements teacher attention-based evidence detection and improved CLIP-based detection
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import re
from transformers import CLIPModel, CLIPProcessor

# Optional imports with fallbacks
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Video processing will be limited.")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("Warning: NLTK not available. Text processing will be limited.")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("Warning: spaCy not available. Advanced text processing will be limited.")

# Download required NLTK data
if HAS_NLTK:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


class AdvancedEvidenceDetector:
    """Advanced evidence detection with multiple methods"""
    
    def __init__(self, method: str = 'clip', config=None):
        self.method = method
        self.config = config
        
        # Initialize CLIP for visual-text similarity
        if method in ['clip', 'hybrid']:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
        
        # Initialize spaCy for better text processing
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None
        
        # Stop words for text processing
        if HAS_NLTK:
            self.stop_words = set(stopwords.words('english'))
        else:
            # Fallback stop words
            self.stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                             'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                             'to', 'was', 'will', 'with', 'would'}
    
    def detect_evidence_timestamp(self, video_path: str, question: str, 
                                manual_timestamp: Optional[float] = None,
                                teacher_model=None) -> float:
        """
        Detect evidence timestamp using the specified method
        
        Args:
            video_path: Path to video file
            question: Question text
            manual_timestamp: Manual annotation if available
            teacher_model: Teacher model for attention-based detection
            
        Returns:
            Evidence timestamp in seconds
        """
        if self.method == 'manual' and manual_timestamp is not None:
            return manual_timestamp
        elif self.method == 'clip':
            return self._detect_clip_evidence(video_path, question)
        elif self.method == 'attention' and teacher_model is not None:
            return self._detect_attention_evidence(video_path, question, teacher_model)
        elif self.method == 'hybrid':
            return self._detect_hybrid_evidence(video_path, question, teacher_model)
        else:
            # Fallback: assume evidence appears at 10% of video length
            return self._get_video_length(video_path) * 0.1
    
    def _detect_clip_evidence(self, video_path: str, question: str) -> float:
        """Enhanced CLIP-based evidence detection"""
        # Extract key visual concepts from question
        visual_keywords = self._extract_visual_concepts(question)
        
        # Process video frames
        frames, timestamps = self._extract_video_frames(video_path)
        if not frames:
            return 0.0
        
        # Compute similarities for each visual concept
        concept_similarities = {}
        
        for concept in visual_keywords:
            if concept.strip():  # Skip empty concepts
                similarities = self._compute_clip_similarities(frames, concept)
                concept_similarities[concept] = similarities
        
        # If no visual concepts found, use the full question
        if not concept_similarities:
            similarities = self._compute_clip_similarities(frames, question)
            concept_similarities['full_question'] = similarities
        
        # Aggregate similarities across concepts
        if len(concept_similarities) == 1:
            final_similarities = list(concept_similarities.values())[0]
        else:
            # Take maximum similarity across concepts for each frame
            all_sims = np.array(list(concept_similarities.values()))
            final_similarities = np.max(all_sims, axis=0)
        
        # Smooth similarities
        final_similarities = self._smooth_similarities(final_similarities)
        
        # Find evidence timestamp
        threshold = np.percentile(final_similarities, 
                                self.config.clip_threshold_percentile if self.config else 90.0)
        
        evidence_indices = np.where(final_similarities >= threshold)[0]
        if len(evidence_indices) > 0:
            return timestamps[evidence_indices[0]]
        else:
            return timestamps[0]
    
    def _detect_attention_evidence(self, video_path: str, question: str, 
                                 teacher_model) -> float:
        """Teacher attention-based evidence detection"""
        # This requires a full-cache teacher model run
        # Implementation depends on your specific model architecture
        
        frames, timestamps = self._extract_video_frames(video_path)
        if not frames:
            return 0.0
        
        try:
            # Run teacher model with full cache
            teacher_model.clear_cache()
            
            # Process video frames
            video_tensor = self._frames_to_tensor(frames)
            teacher_model.encode_video(video_tensor)
            
            # Get attention weights for the question
            attention_weights = self._extract_attention_weights(teacher_model, question)
            
            # Aggregate attention over frames
            frame_attention = self._aggregate_frame_attention(attention_weights, len(frames))
            
            # Find evidence timestamp
            threshold = np.percentile(frame_attention, 
                                    self.config.attention_threshold_percentile if self.config else 95.0)
            
            evidence_indices = np.where(frame_attention >= threshold)[0]
            if len(evidence_indices) > 0:
                return timestamps[evidence_indices[0]]
            else:
                return timestamps[0]
        
        except Exception as e:
            print(f"Attention-based detection failed: {e}")
            # Fallback to CLIP
            return self._detect_clip_evidence(video_path, question)
    
    def _detect_hybrid_evidence(self, video_path: str, question: str, 
                              teacher_model=None) -> float:
        """Hybrid approach combining CLIP and attention"""
        clip_timestamp = self._detect_clip_evidence(video_path, question)
        
        if teacher_model is not None:
            attention_timestamp = self._detect_attention_evidence(video_path, question, teacher_model)
            # Average the two timestamps (could use other combination strategies)
            return (clip_timestamp + attention_timestamp) / 2.0
        else:
            return clip_timestamp
    
    def _extract_visual_concepts(self, question: str) -> List[str]:
        """Extract visual concepts from question text"""
        visual_keywords = []
        
        if self.nlp is not None:
            # Use spaCy for better entity and noun extraction
            doc = self.nlp(question)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    visual_keywords.append(ent.text.lower())
            
            # Extract nouns and adjectives
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and 
                    token.text.lower() not in self.stop_words and
                    len(token.text) > 2):
                    visual_keywords.append(token.text.lower())
        
        else:
            # Fallback to simple tokenization
            if HAS_NLTK:
                tokens = word_tokenize(question.lower())
            else:
                # Very basic tokenization
                tokens = re.findall(r'\b\w+\b', question.lower())
            
            visual_keywords = [token for token in tokens 
                             if token not in self.stop_words and 
                             len(token) > 2 and 
                             token.isalpha()]
        
        # Add some domain-specific visual terms
        visual_terms = [
            'color', 'red', 'blue', 'green', 'yellow', 'black', 'white',
            'person', 'people', 'man', 'woman', 'child', 'car', 'vehicle',
            'animal', 'dog', 'cat', 'bird', 'house', 'building', 'room',
            'table', 'chair', 'door', 'window', 'food', 'drink', 'clothing'
        ]
        
        for term in visual_terms:
            if term in question.lower() and term not in visual_keywords:
                visual_keywords.append(term)
        
        return list(set(visual_keywords))  # Remove duplicates
    
    def _extract_video_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """Extract frames from video at specified FPS"""
        if not HAS_CV2:
            print("OpenCV not available. Cannot extract video frames.")
            return [], []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return [], []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames
        sample_fps = self.config.sample_fps if self.config else 0.5
        sample_interval = max(1, int(fps / sample_fps))
        
        frames = []
        timestamps = []
        
        frame_idx = 0
        while frame_idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(frame_idx / fps)
            
            frame_idx += sample_interval
        
        cap.release()
        return frames, timestamps
    
    def _compute_clip_similarities(self, frames: List[np.ndarray], text: str) -> np.ndarray:
        """Compute CLIP similarities between frames and text"""
        if not frames:
            return np.array([])
        
        # Process in batches to avoid memory issues
        batch_size = 8
        similarities = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            inputs = self.clip_processor(
                text=[text] * len(batch_frames),
                images=batch_frames,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                batch_similarities = outputs.logits_per_text[0].cpu().numpy()
                similarities.extend(batch_similarities)
        
        return np.array(similarities)
    
    def _smooth_similarities(self, similarities: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Smooth similarities with temporal kernel"""
        if len(similarities) < kernel_size:
            return similarities
        
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(similarities, kernel, mode='same')
        return smoothed
    
    def _get_video_length(self, video_path: str) -> float:
        """Get video length in seconds"""
        if not HAS_CV2:
            print("OpenCV not available. Cannot get video length.")
            return 60.0  # Fallback to 1 minute
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = frame_count / fps
        cap.release()
        return length
    
    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert frames to tensor format expected by model"""
        # This is a placeholder - implement according to your model's input format
        frame_tensors = []
        for frame in frames:
            # Resize and normalize frame
            if HAS_CV2:
                frame_resized = cv2.resize(frame, (224, 224))
            else:
                # Fallback: assume frame is already correct size or use numpy
                frame_resized = frame
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
            frame_tensors.append(frame_tensor)
        
        return torch.stack(frame_tensors)
    
    def _extract_attention_weights(self, model, question: str) -> np.ndarray:
        """Extract attention weights from teacher model"""
        # This is a placeholder - implement according to your model's architecture
        # You would need to:
        # 1. Tokenize the question
        # 2. Run forward pass with attention output
        # 3. Extract cross-attention weights between question tokens and vision tokens
        
        # For now, return random weights as placeholder
        return np.random.rand(100, 196)  # (seq_len, vision_tokens)
    
    def _aggregate_frame_attention(self, attention_weights: np.ndarray, 
                                 num_frames: int) -> np.ndarray:
        """Aggregate attention weights over frames"""
        # This depends on how vision tokens are organized
        # Assuming vision tokens are organized as [frame1_tokens, frame2_tokens, ...]
        
        tokens_per_frame = attention_weights.shape[1] // num_frames
        frame_attention = []
        
        for i in range(num_frames):
            start_idx = i * tokens_per_frame
            end_idx = (i + 1) * tokens_per_frame
            frame_tokens = attention_weights[:, start_idx:end_idx]
            
            # Average attention over tokens and sequence positions
            frame_attn = np.mean(frame_tokens)
            frame_attention.append(frame_attn)
        
        return np.array(frame_attention)


class EvidenceAnnotationTool:
    """Tool for creating manual evidence annotations"""
    
    def __init__(self, video_dir: str, questions_file: str):
        self.video_dir = video_dir
        self.questions_file = questions_file
        
    def annotate_evidence(self, output_file: str):
        """Interactive tool for annotating evidence timestamps"""
        with open(self.questions_file, 'r') as f:
            questions = json.load(f)
        
        annotations = {}
        
        for i, q in enumerate(questions):
            print(f"\n--- Question {i+1}/{len(questions)} ---")
            print(f"Video: {q['video_id']}")
            print(f"Question: {q['question']}")
            
            video_path = os.path.join(self.video_dir, f"{q['video_id']}.mp4")
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue
            
            # Play video (you could use opencv or other video player)
            print(f"Please watch the video and identify when evidence first appears.")
            
            # Get user input
            while True:
                try:
                    timestamp = float(input("Enter evidence timestamp (seconds): "))
                    break
                except ValueError:
                    print("Please enter a valid number.")
            
            annotations[q.get('id', i)] = {
                'video_id': q['video_id'],
                'question': q['question'],
                'evidence_timestamp': timestamp
            }
        
        # Save annotations
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Annotations saved to {output_file}")


def validate_evidence_detection(detector: AdvancedEvidenceDetector,
                              video_questions: List[Dict],
                              video_dir: str,
                              manual_annotations: Dict = None) -> Dict:
    """Validate evidence detection against manual annotations"""
    
    if manual_annotations is None:
        print("No manual annotations provided for validation")
        return {}
    
    results = {
        'method': detector.method,
        'mae': [],  # Mean Absolute Error
        'correlations': [],
        'agreements': []  # Within X seconds agreement
    }
    
    for vq in video_questions:
        qid = vq.get('id')
        if qid not in manual_annotations:
            continue
        
        video_path = os.path.join(video_dir, f"{vq['video_id']}.mp4")
        if not os.path.exists(video_path):
            continue
        
        manual_timestamp = manual_annotations[qid]['evidence_timestamp']
        detected_timestamp = detector.detect_evidence_timestamp(video_path, vq['question'])
        
        # Compute metrics
        mae = abs(detected_timestamp - manual_timestamp)
        results['mae'].append(mae)
        
        # Agreement within 30 seconds
        agreement = mae <= 30.0
        results['agreements'].append(agreement)
    
    # Summary statistics
    if results['mae']:
        results['mean_mae'] = np.mean(results['mae'])
        results['std_mae'] = np.std(results['mae'])
        results['agreement_rate'] = np.mean(results['agreements'])
    
    return results


if __name__ == "__main__":
    # Example usage
    from qar_measurement import QARConfig
    
    config = QARConfig()
    detector = AdvancedEvidenceDetector(method='clip', config=config)
    
    # Test on a single video-question pair
    video_path = "path/to/test_video.mp4"
    question = "What color is the car?"
    
    if os.path.exists(video_path):
        timestamp = detector.detect_evidence_timestamp(video_path, question)
        print(f"Evidence timestamp: {timestamp:.2f} seconds")
    else:
        print("Test video not found")