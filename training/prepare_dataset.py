"""
Dataset preparation for Face Swap Detection
Supports Celeb-DF v2, FaceForensics++, and other standard datasets
Creates video-disjoint and identity-disjoint splits
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split


class FaceSwapDatasetPreparator:
    """
    Prepares face swap datasets for training with proper splits:
    - Video-disjoint: No video appears in both train and test
    - Identity-disjoint: No identity appears in both train and test
    """
    
    def __init__(self, 
                 data_root: str,
                 output_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        random.seed(seed)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scan_celeb_df_v2(self, dataset_path: str) -> List[Dict]:
        """
        Scan Celeb-DF v2 dataset structure
        Expected structure:
        dataset_path/
            Celeb-real/
                id00001/
                    video1.mp4
                    ...
            Celeb-synthesis/
                id00001/
                    video1.mp4
                    ...
            YouTube-real/
                video1.mp4
                ...
        """
        dataset_path = Path(dataset_path)
        videos = []
        
        # Real videos
        real_dirs = ['Celeb-real', 'YouTube-real']
        for real_dir in real_dirs:
            real_path = dataset_path / real_dir
            if real_path.exists():
                for video_file in real_path.rglob('*.mp4'):
                    # Extract identity if possible
                    rel_path = video_file.relative_to(dataset_path)
                    parts = rel_path.parts
                    identity = parts[1] if len(parts) > 1 else 'unknown'
                    
                    videos.append({
                        'path': str(video_file),
                        'label': 0,  # REAL
                        'dataset': 'celeb_df_v2',
                        'subset': real_dir,
                        'identity': identity,
                        'video_name': video_file.stem
                    })
        
        # Fake/SwapFace videos
        fake_path = dataset_path / 'Celeb-synthesis'
        if fake_path.exists():
            for video_file in fake_path.rglob('*.mp4'):
                rel_path = video_file.relative_to(dataset_path)
                parts = rel_path.parts
                identity = parts[1] if len(parts) > 1 else 'unknown'
                
                videos.append({
                    'path': str(video_file),
                    'label': 1,  # FAKE/SWAPFACE
                    'dataset': 'celeb_df_v2',
                    'subset': 'Celeb-synthesis',
                    'identity': identity,
                    'video_name': video_file.stem
                })
        
        return videos
    
    def scan_faceforensics_pp(self, dataset_path: str) -> List[Dict]:
        """
        Scan FaceForensics++ dataset structure
        Expected structure:
        dataset_path/
            original_sequences/
                youtube/
                    c23/
                        videos/
                            000_001.mp4
            manipulated_sequences/
                Deepfakes/
                    c23/
                        videos/
                            000_001.mp4
                Face2Face/
                    c23/
                        videos/
                            000_001.mp4
                FaceSwap/
                    c23/
                        videos/
                            000_001.mp4
                NeuralTextures/
                    c23/
                        videos/
                            000_001.mp4
        """
        dataset_path = Path(dataset_path)
        videos = []
        
        # Real videos
        original_path = dataset_path / 'original_sequences' / 'youtube' / 'c23' / 'videos'
        if original_path.exists():
            for video_file in original_path.glob('*.mp4'):
                videos.append({
                    'path': str(video_file),
                    'label': 0,  # REAL
                    'dataset': 'faceforensics++',
                    'subset': 'original',
                    'method': 'original',
                    'identity': video_file.stem.split('_')[0],  # e.g., 000
                    'video_name': video_file.stem
                })
        
        # Fake videos - focus on FaceSwap for our use case
        manipulated_path = dataset_path / 'manipulated_sequences'
        if manipulated_path.exists():
            for method_dir in manipulated_path.iterdir():
                if not method_dir.is_dir():
                    continue
                
                method_name = method_dir.name
                # Prioritize FaceSwap but include others for generalization
                c23_path = method_dir / 'c23' / 'videos'
                if c23_path.exists():
                    for video_file in c23_path.glob('*.mp4'):
                        videos.append({
                            'path': str(video_file),
                            'label': 1,  # FAKE
                            'dataset': 'faceforensics++',
                            'subset': 'manipulated',
                            'method': method_name.lower(),
                            'identity': video_file.stem.split('_')[0],
                            'video_name': video_file.stem
                        })
        
        return videos
    
    def scan_generic_dataset(self, dataset_path: str, dataset_name: str) -> List[Dict]:
        """
        Scan generic dataset with real/fake folder structure
        Expected structure:
        dataset_path/
            real/
                video1.mp4
                ...
            fake/
                video1.mp4
                ...
        """
        dataset_path = Path(dataset_path)
        videos = []
        
        real_path = dataset_path / 'real'
        if real_path.exists():
            for video_file in real_path.rglob('*.mp4'):
                videos.append({
                    'path': str(video_file),
                    'label': 0,
                    'dataset': dataset_name,
                    'subset': 'real',
                    'identity': 'unknown',
                    'video_name': video_file.stem
                })
        
        fake_path = dataset_path / 'fake'
        if fake_path.exists():
            for video_file in fake_path.rglob('*.mp4'):
                videos.append({
                    'path': str(video_file),
                    'label': 1,
                    'dataset': dataset_name,
                    'subset': 'fake',
                    'identity': 'unknown',
                    'video_name': video_file.stem
                })
        
        return videos
    
    def create_splits(self, videos: List[Dict], split_type: str = 'video_disjoint') -> Dict[str, List[Dict]]:
        """
        Create train/val/test splits
        split_type: 'video_disjoint' or 'identity_disjoint'
        """
        if split_type == 'video_disjoint':
            return self._video_disjoint_split(videos)
        elif split_type == 'identity_disjoint':
            return self._identity_disjoint_split(videos)
        else:
            raise ValueError(f"Unknown split type: {split_type}")
    
    def _video_disjoint_split(self, videos: List[Dict]) -> Dict[str, List[Dict]]:
        """Split by video - no video overlap between splits"""
        # Group by dataset to maintain balance
        by_dataset = defaultdict(list)
        for v in videos:
            by_dataset[v['dataset']].append(v)
        
        splits = {'train': [], 'val': [], 'test': []}
        
        for dataset_name, dataset_videos in by_dataset.items():
            # Stratify by label
            real_videos = [v for v in dataset_videos if v['label'] == 0]
            fake_videos = [v for v in dataset_videos if v['label'] == 1]
            
            for label_videos in [real_videos, fake_videos]:
                if len(label_videos) == 0:
                    continue
                
                # Shuffle
                random.shuffle(label_videos)
                
                n_total = len(label_videos)
                n_train = int(n_total * self.train_ratio)
                n_val = int(n_total * self.val_ratio)
                
                splits['train'].extend(label_videos[:n_train])
                splits['val'].extend(label_videos[n_train:n_train + n_val])
                splits['test'].extend(label_videos[n_train + n_val:])
        
        # Shuffle each split
        for split in splits.values():
            random.shuffle(split)
        
        return splits
    
    def _identity_disjoint_split(self, videos: List[Dict]) -> Dict[str, List[Dict]]:
        """Split by identity - no identity overlap between splits"""
        # Group by identity
        by_identity = defaultdict(list)
        for v in videos:
            identity = v.get('identity', 'unknown')
            by_identity[identity].append(v)
        
        identities = list(by_identity.keys())
        random.shuffle(identities)
        
        n_total = len(identities)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_identities = identities[:n_train]
        val_identities = identities[n_train:n_train + n_val]
        test_identities = identities[n_train + n_val:]
        
        splits = {'train': [], 'val': [], 'test': []}
        
        for identity, identity_videos in by_identity.items():
            if identity in train_identities:
                splits['train'].extend(identity_videos)
            elif identity in val_identities:
                splits['val'].extend(identity_videos)
            else:
                splits['test'].extend(identity_videos)
        
        # Shuffle each split
        for split in splits.values():
            random.shuffle(split)
        
        return splits
    
    def save_splits(self, splits: Dict[str, List[Dict]], prefix: str = ''):
        """Save splits to JSON files"""
        for split_name, split_videos in splits.items():
            output_file = self.output_dir / f"{prefix}{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(split_videos, f, indent=2)
            print(f"Saved {len(split_videos)} videos to {output_file}")
        
        # Save summary
        summary = {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test']),
            'total': sum(len(v) for v in splits.values()),
            'split_type': 'video_disjoint'  # or identity_disjoint
        }
        with open(self.output_dir / f"{prefix}summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def prepare_all(self, datasets_config: List[Dict]):
        """
        Prepare all datasets
        datasets_config: List of dicts with keys:
            - 'path': path to dataset
            - 'name': dataset name
            - 'type': 'celeb_df_v2', 'faceforensics++', or 'generic'
        """
        all_videos = []
        
        for ds_config in datasets_config:
            path = ds_config['path']
            name = ds_config['name']
            dtype = ds_config.get('type', 'generic')
            
            print(f"Scanning {name} ({dtype}) from {path}...")
            
            if dtype == 'celeb_df_v2':
                videos = self.scan_celeb_df_v2(path)
            elif dtype == 'faceforensics++':
                videos = self.scan_faceforensics_pp(path)
            else:
                videos = self.scan_generic_dataset(path, name)
            
            print(f"  Found {len(videos)} videos")
            all_videos.extend(videos)
        
        print(f"Total videos: {len(all_videos)}")
        
        # Create video-disjoint splits (primary)
        print("Creating video-disjoint splits...")
        video_splits = self.create_splits(all_videos, 'video_disjoint')
        self.save_splits(video_splits, 'video_disjoint_')
        
        # Create identity-disjoint splits (for cross-dataset evaluation)
        print("Creating identity-disjoint splits...")
        identity_splits = self.create_splits(all_videos, 'identity_disjoint')
        self.save_splits(identity_splits, 'identity_disjoint_')
        
        # Save combined metadata
        metadata = {
            'total_videos': len(all_videos),
            'real_videos': sum(1 for v in all_videos if v['label'] == 0),
            'fake_videos': sum(1 for v in all_videos if v['label'] == 1),
            'datasets': list(set(v['dataset'] for v in all_videos)),
            'video_disjoint': {
                'train': len(video_splits['train']),
                'val': len(video_splits['val']),
                'test': len(video_splits['test'])
            },
            'identity_disjoint': {
                'train': len(identity_splits['train']),
                'val': len(identity_splits['val']),
                'test': len(identity_splits['test'])
            }
        }
        
        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return video_splits, identity_splits


def create_deepfakebench_json(splits: Dict[str, List[Dict]], output_dir: str, dataset_name: str):
    """Create JSON files compatible with DeepfakeBench format"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, videos in splits.items():
        # DeepfakeBench expects: {dataset_name: {label: {split: {video_name: {frames: [...]}}}}}
        data = {dataset_name: {'REAL': {split_name: {}}, 'FAKE': {split_name: {}}}}
        
        for v in videos:
            label_str = 'REAL' if v['label'] == 0 else 'FAKE'
            video_key = v['video_name']
            
            # For frame-level, we'd need extracted frames
            # For now, store video path
            data[dataset_name][label_str][split_name][video_key] = {
                'frames': [v['path']],  # Placeholder - would be frame paths
                'label': v['label']
            }
        
        output_file = output_path / f"{dataset_name}_{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Created DeepfakeBench compatible JSON: {output_file}")


if __name__ == "__main__":
    preparator = FaceSwapDatasetPreparator(
        data_root="data/",
        output_dir="data/splits/",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    datasets_config = [
        {'path': 'data/Celeb-DF-v2', 'name': 'celeb_df_v2', 'type': 'celeb_df_v2'},
    ]
    
    print("Preparing Celeb-DF-v2 dataset splits...")
    video_splits, identity_splits = preparator.prepare_all(datasets_config)
    print("Done!")