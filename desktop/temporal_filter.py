"""
Temporal Smoothing Module for Stable Realtime Predictions
Implements EMA, Sliding Window, and Voting methods with hysteresis
"""

import numpy as np
from collections import deque
from typing import Deque, List, Optional, Tuple
from enum import Enum
import logging


class SignalState(Enum):
    """Signal state after temporal filtering"""
    REAL = "REAL"
    SUSPICIOUS = "SUSPICIOUS"
    FAKE = "FAKE"
    UNKNOWN = "UNKNOWN"


class TemporalFilter:
    """Base class for temporal filters"""
    
    def __init__(self, hysteresis_high: float = 0.7, hysteresis_low: float = 0.3):
        self.hysteresis_high = hysteresis_high
        self.hysteresis_low = hysteresis_low
        self.current_state = SignalState.UNKNOWN
        self.logger = logging.getLogger(__name__)
    
    def update(self, fake_score: float, quality: str = 'GOOD') -> Tuple[float, SignalState]:
        """
        Update filter with new prediction
        
        Returns:
            (smoothed_score, signal_state)
        """
        raise NotImplementedError
    
    def _apply_hysteresis(self, score: float) -> SignalState:
        """Apply hysteresis to determine state"""
        if self.current_state in [SignalState.FAKE, SignalState.SUSPICIOUS]:
            # Currently suspicious/fake - need score to drop below low threshold
            if score <= self.hysteresis_low:
                self.current_state = SignalState.REAL
            elif score >= self.hysteresis_high:
                self.current_state = SignalState.FAKE
            else:
                self.current_state = SignalState.SUSPICIOUS
        else:
            # Currently real - need score to exceed high threshold
            if score >= self.hysteresis_high:
                self.current_state = SignalState.FAKE
            elif score > self.hysteresis_low:
                self.current_state = SignalState.SUSPICIOUS
            else:
                self.current_state = SignalState.REAL
        
        return self.current_state
    
    def reset(self):
        """Reset filter state"""
        self.current_state = SignalState.UNKNOWN


class EMATemporalFilter(TemporalFilter):
    """Exponential Moving Average temporal filter"""
    
    def __init__(self, 
                 alpha: float = 0.3,
                 hysteresis_high: float = 0.7,
                 hysteresis_low: float = 0.3):
        super().__init__(hysteresis_high, hysteresis_low)
        self.alpha = alpha
        self.smoothed_score = None
        self.initialized = False
    
    def update(self, fake_score: float, quality: str = 'GOOD') -> Tuple[float, SignalState]:
        if quality != 'GOOD':
            # Don't update on poor quality frames
            if self.smoothed_score is not None:
                state = self._apply_hysteresis(self.smoothed_score)
                return self.smoothed_score, state
            return 0.0, SignalState.UNKNOWN
        
        if not self.initialized:
            self.smoothed_score = fake_score
            self.initialized = True
        else:
            # EMA: S_t = α * S_{t-1} + (1-α) * P_t
            self.smoothed_score = (self.alpha * self.smoothed_score + 
                                   (1 - self.alpha) * fake_score)
        
        state = self._apply_hysteresis(self.smoothed_score)
        return self.smoothed_score, state
    
    def reset(self):
        super().reset()
        self.smoothed_score = None
        self.initialized = False


class SlidingWindowFilter(TemporalFilter):
    """Sliding window average/median filter"""
    
    def __init__(self, 
                 window_size: int = 5,
                 method: str = 'mean',  # 'mean' or 'median'
                 hysteresis_high: float = 0.7,
                 hysteresis_low: float = 0.3):
        super().__init__(hysteresis_high, hysteresis_low)
        self.window_size = window_size
        self.method = method
        self.window: Deque[float] = deque(maxlen=window_size)
    
    def update(self, fake_score: float, quality: str = 'GOOD') -> Tuple[float, SignalState]:
        if quality == 'GOOD':
            self.window.append(fake_score)
        
        if len(self.window) == 0:
            return 0.0, SignalState.UNKNOWN
        
        if self.method == 'median':
            smoothed_score = float(np.median(self.window))
        else:
            smoothed_score = float(np.mean(self.window))
        
        state = self._apply_hysteresis(smoothed_score)
        return smoothed_score, state
    
    def reset(self):
        super().reset()
        self.window.clear()


class VotingFilter(TemporalFilter):
    """Temporal voting filter - requires N consecutive predictions"""
    
    def __init__(self,
                 window_size: int = 5,
                 min_positive_votes: int = 3,
                 hysteresis_high: float = 0.7,
                 hysteresis_low: float = 0.3):
        super().__init__(hysteresis_high, hysteresis_low)
        self.window_size = window_size
        self.min_positive_votes = min_positive_votes
        self.window: Deque[int] = deque(maxlen=window_size)  # 1 for fake, 0 for real
    
    def update(self, fake_score: float, quality: str = 'GOOD') -> Tuple[float, SignalState]:
        if quality == 'GOOD':
            vote = 1 if fake_score >= self.hysteresis_high else 0
            self.window.append(vote)
        
        if len(self.window) == 0:
            return 0.0, SignalState.UNKNOWN
        
        positive_votes = sum(self.window)
        total_votes = len(self.window)
        smoothed_score = positive_votes / total_votes
        
        # Require minimum votes for positive decision
        if positive_votes >= self.min_positive_votes:
            self.current_state = SignalState.FAKE
        elif positive_votes == 0:
            self.current_state = SignalState.REAL
        else:
            self.current_state = SignalState.SUSPICIOUS
        
        return smoothed_score, self.current_state
    
    def reset(self):
        super().reset()
        self.window.clear()


class MultiTrackTemporalFilter:
    """Temporal filter for multiple tracked faces"""
    
    def __init__(self, filter_type: str = 'ema', **kwargs):
        self.filter_type = filter_type
        self.filter_kwargs = kwargs
        self.track_filters: dict = {}  # track_id -> TemporalFilter
        self.logger = logging.getLogger(__name__)
    
    def _create_filter(self) -> TemporalFilter:
        """Create new filter instance"""
        if self.filter_type == 'ema':
            return EMATemporalFilter(**self.filter_kwargs)
        elif self.filter_type == 'sliding_window':
            return SlidingWindowFilter(**self.filter_kwargs)
        elif self.filter_type == 'voting':
            return VotingFilter(**self.filter_kwargs)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
    
    def update(self, track_id: int, fake_score: float, quality: str = 'GOOD') -> Tuple[float, SignalState]:
        """Update filter for specific track"""
        if track_id not in self.track_filters:
            self.track_filters[track_id] = self._create_filter()
        
        return self.track_filters[track_id].update(fake_score, quality)
    
    def remove_track(self, track_id: int):
        """Remove filter for track"""
        if track_id in self.track_filters:
            del self.track_filters[track_id]
    
    def cleanup(self, active_track_ids: List[int]):
        """Remove filters for inactive tracks"""
        to_remove = [tid for tid in self.track_filters if tid not in active_track_ids]
        for tid in to_remove:
            self.remove_track(tid)
    
    def reset_all(self):
        """Reset all track filters"""
        for f in self.track_filters.values():
            f.reset()
        self.track_filters.clear()


def create_temporal_filter(config: dict) -> TemporalFilter:
    """Create temporal filter from config"""
    filter_config = config.get('temporal_smoothing', {})
    method = filter_config.get('method', 'ema')
    hysteresis_high = filter_config.get('hysteresis_high', 0.7)
    hysteresis_low = filter_config.get('hysteresis_low', 0.3)
    
    if method == 'ema':
        return EMATemporalFilter(
            alpha=filter_config.get('alpha', 0.3),
            hysteresis_high=hysteresis_high,
            hysteresis_low=hysteresis_low
        )
    elif method == 'sliding_window':
        return SlidingWindowFilter(
            window_size=filter_config.get('window_size', 5),
            method=filter_config.get('window_method', 'mean'),
            hysteresis_high=hysteresis_high,
            hysteresis_low=hysteresis_low
        )
    elif method == 'voting':
        return VotingFilter(
            window_size=filter_config.get('window_size', 5),
            min_positive_votes=filter_config.get('min_votes', 3),
            hysteresis_high=hysteresis_high,
            hysteresis_low=hysteresis_low
        )
    else:
        raise ValueError(f"Unknown temporal filter method: {method}")


def create_multi_track_filter(config: dict) -> MultiTrackTemporalFilter:
    """Create multi-track temporal filter from config"""
    filter_config = config.get('temporal_smoothing', {})
    method = filter_config.get('method', 'ema')
    
    return MultiTrackTemporalFilter(
        filter_type=method,
        **{k: v for k, v in filter_config.items() if k != 'method'}
    )


if __name__ == "__main__":
    # Test temporal filters
    logging.basicConfig(level=logging.INFO)
    
    # Simulate fake scores
    scores = [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 
              0.85, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
    
    print("Testing EMA Filter:")
    ema = EMATemporalFilter(alpha=0.3, hysteresis_high=0.7, hysteresis_low=0.3)
    for s in scores:
        smoothed, state = ema.update(s)
        print(f"  Input: {s:.2f} -> Smoothed: {smoothed:.2f} -> State: {state.value}")
    
    print("\nTesting Sliding Window Filter:")
    sw = SlidingWindowFilter(window_size=5, method='mean', hysteresis_high=0.7, hysteresis_low=0.3)
    for s in scores:
        smoothed, state = sw.update(s)
        print(f"  Input: {s:.2f} -> Smoothed: {smoothed:.2f} -> State: {state.value}")
    
    print("\nTesting Voting Filter:")
    vf = VotingFilter(window_size=5, min_positive_votes=3, hysteresis_high=0.7, hysteresis_low=0.3)
    for s in scores:
        smoothed, state = vf.update(s)
        print(f"  Input: {s:.2f} -> Smoothed: {smoothed:.2f} -> State: {state.value}")