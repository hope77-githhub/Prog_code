"""
이벤트 특성 추출을 위한 Feature_Extractor_event 클래스
"""

import math
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm
from Research.project.Analysis.config import SPEAKERS

class Feature_Extractor_event:
    def __init__(self, event_data: List[Dict]):
        """
        이벤트 기준 슬라이싱된 데이터 리스트를 받아 피처를 추출합니다.
        
        Args:
            event_data (list of dict): 각 이벤트 데이터 예시:
                {
                    'event_type': 'TURRET_PLATE_DESTROYED',
                    'gold': 125,
                    'success': 1,
                    'time': 899481,
                    'sequence': ['t', 'a', 'j', 's', ...],
                    'start_time': [...],
                    'end_time': [...]
                }
        """
        self.event_data = event_data
        self.speakers = SPEAKERS  # 기준 발화자 순서 저장

    def compute_entropy(self, seq: List[str]) -> float:
        """
        주어진 sequence의 엔트로피 계산
        
        Args:
            seq: 시퀀스 리스트
            
        Returns:
            float: 계산된 엔트로피 값
        """
        if not seq:
            return 0.0
        count = Counter(seq)
        total = len(seq)
        entropy = 0.0
        for freq in count.values():
            p = freq / total
            entropy -= p * math.log2(p)
        return round(entropy, 3)

    def compute_lsa_z_scores(self, seq: List[str]) -> Dict[str, float]:
        """
        길이 2 전이(2-gram)에 대한 Z-score를 계산합니다.
        
        Args:
            seq: 시퀀스 리스트
            
        Returns:
            dict: 각 2-gram에 대한 Z-score 값
        """
        lsa_z = {}
        U = len(seq)
        if U < 2:
            tokens = set(seq)
            for t1 in tokens:
                for t2 in tokens:
                    key = f"{t1}{t2}"
                    lsa_z[key] = 0.0
            return lsa_z

        T = U - 1
        row_counts = Counter(seq[:-1])
        col_counts = Counter(seq[1:])
        pair_counts = {}
        for i in range(T):
            key = f"{seq[i]}{seq[i+1]}"
            pair_counts[key] = pair_counts.get(key, 0) + 1

        def p_row(token: str) -> float:
            return row_counts[token] / T if T > 0 else 0

        def p_col(token: str) -> float:
            return col_counts[token] / T if T > 0 else 0

        for key, observed in pair_counts.items():
            token1, token2 = key[0], key[1]
            expected = (row_counts[token1] * col_counts[token2]) / T if T > 0 else 0
            pGplus = p_row(token1)
            pPlusT = p_col(token2)
            denom = expected * (1 - pGplus) * (1 - pPlusT)
            if denom > 0:
                z_score = (observed - expected) / math.sqrt(denom)
                lsa_z[key] = round(z_score, 3)
            else:
                lsa_z[key] = 0.0

        # 모든 가능한 화자 조합에 대해 Z-score 생성
        for t1 in self.speakers:
            for t2 in self.speakers:
                check_key = f"{t1}{t2}"
                if check_key not in lsa_z:
                    lsa_z[check_key] = 0.0

        return lsa_z

    def extract_ngrams(self, seq: List[str]) -> Tuple[Dict[Tuple[str, ...], int], Dict[Tuple[str, ...], int]]:
        """
        시퀀스에서 2-gram과 3-gram을 추출합니다.
        
        Args:
            seq: 시퀀스 리스트
            
        Returns:
            tuple: (2-gram 딕셔너리, 3-gram 딕셔너리)
        """
        if len(seq) < 2:
            return {}, {}
            
        ngram2 = {}
        for i in range(len(seq) - 1):
            key = tuple(seq[i:i+2])
            ngram2[key] = ngram2.get(key, 0) + 1
            
        ngram3 = {}
        if len(seq) >= 3:
            for i in range(len(seq) - 2):
                key = tuple(seq[i:i+3])
                ngram3[key] = ngram3.get(key, 0) + 1
                
        return ngram2, ngram3

    # T-pattern 관련 함수들 --------------------------------------------
    def get_repeated_indices(self, seq: List[str],
                             pattern: Tuple[str, ...]) -> List[Tuple[int, int]]:
        """
        시퀀스에서 주어진 패턴이 반복되는 인덱스 위치를 찾습니다.
        
        Args:
            seq: 시퀀스 리스트
            pattern: 찾을 패턴 튜플
            
        Returns:
            list: 패턴이 발견된 (시작, 끝) 인덱스 튜플 리스트
        """
        repeated = []
        p_len = len(pattern)
        for i in range(len(seq) - p_len + 1):
            if tuple(seq[i:i+p_len]) == pattern:
                repeated.append((i, i + p_len - 1))
        return repeated

    def filter_significant_indices(self, indices: List[Tuple[int, int]],
                                   start_times: List[int],
                                   end_times: List[int],
                                   CI: int) -> List[Tuple[int, int]]:
        """
        Critical Interval(CI) 조건을 만족하는 인덱스만 필터링합니다.
        
        Args:
            indices: (시작, 끝) 인덱스 튜플 리스트
            start_times: 시작 시간 리스트
            end_times: 종료 시간 리스트
            CI: Critical Interval (밀리초)
            
        Returns:
            list: CI 조건을 만족하는 인덱스 튜플 리스트
        """
        filtered = []
        for (s_idx, e_idx) in indices:
            if s_idx < e_idx < len(start_times):
                is_valid = True
                for k in range(s_idx, e_idx):
                    # if you want to use end time, start_time => end time 
                    gap = start_times[k+1] - end_times[k]
                    if gap > CI:
                        is_valid = False
                        break
                if is_valid:
                    filtered.append((s_idx, e_idx))
            elif s_idx == e_idx:
                filtered.append((s_idx, e_idx))
        return filtered

    def find_patterns_recursive(self, seq: List[str],
                                start_times: List[int],
                                end_times: List[int],
                                CI: int,
                                pattern: Tuple[str, ...],
                                start_index_list: List[Tuple[int, int]],
                                results: List[Dict]) -> None:
        """
        패턴을 재귀적으로 확장하여 찾습니다.
        
        Args:
            seq: 시퀀스 리스트
            start_times: 시작 시간 리스트
            end_times: 종료 시간 리스트
            CI: Critical Interval (밀리초)
            pattern: 현재 패턴 튜플
            start_index_list: 패턴이 발견된 인덱스 리스트
            results: 결과를 저장할 리스트
        """
        # 인접 중복 토큰이 있으면 해당 패턴은 추출하지 않음
        if any(pattern[i] == pattern[i+1] for i in range(len(pattern) - 1)):
            return

        # 현재 패턴을 결과에 추가 (나중에 존재하는 패턴이면 count에 누적할 예정)
        results.append({"pattern": pattern, "indices": start_index_list})
        
        # 패턴 길이가 3이면 더 이상 확장하지 않음
        if len(pattern) == 3:
            return
        
        seq_len = len(seq)
        extended_candidates = {}
        for (s_idx, e_idx) in start_index_list:
            next_idx = e_idx + 1
            if next_idx < seq_len:
                next_token = seq[next_idx]
                new_pattern = pattern + (next_token,)
                extended_candidates.setdefault(new_pattern, [])
                extended_candidates[new_pattern].append((s_idx, next_idx))
        for cand_pat, cand_idx_list in extended_candidates.items():
            rep_indices = self.get_repeated_indices(seq, cand_pat)
            filtered = self.filter_significant_indices(rep_indices, start_times, end_times, CI)
            if filtered:
                self.find_patterns_recursive(seq, start_times, end_times, CI, cand_pat, filtered, results)

    def extract_t_patterns(self, CI: int) -> List[Dict]:
        """
        각 이벤트 데이터에 대해 최대 길이 3까지의 t-pattern을 추출합니다.
        
        Args:
            CI: Critical Interval (밀리초)
            
        Returns:
            list: 이벤트별 t-pattern 추출 결과
        """
        all_results = []
        for event in tqdm(self.event_data, desc="Processing Events for T-patterns"):
            seq = event.get("sequence", [])
            start_times = event.get("start_time", [])
            end_times = event.get("end_time", [])
            event_type = event.get("event_type", "unknown")
            event_time = event.get("time")
            
            event_patterns = {}  # 결과: {pattern: count, ...}
            i = 0
            # while 루프를 사용하여 추출된 패턴 이후부터 다시 탐색
            while i < len(seq) - 1:
                # 2-gram 패턴 추출 (동일 토큰이면 건너뛰기)
                if i + 1 >= len(seq):
                    break
                    
                pattern_2 = tuple(seq[i:i+2])
                if pattern_2[0] == pattern_2[1]:
                    i += 1
                    continue
                    
                rep_indices = self.get_repeated_indices(seq, pattern_2)
                filtered_indices = self.filter_significant_indices(rep_indices, start_times, end_times, CI)
                
                if not filtered_indices:
                    i += 1
                    continue
                    
                temp_results = []
                self.find_patterns_recursive(seq, start_times, end_times, CI, pattern_2, filtered_indices, temp_results)
                
                if not temp_results:
                    i += 1
                    continue
                    
                # temp_results에는 2-gram과 확장 가능한 경우 3-gram 패턴이 포함됨.
                # 추출된 결과 중 최대 길이의 패턴을 선택
                best_pattern = max(temp_results, key=lambda d: len(d["pattern"]))["pattern"]
                # 해당 패턴의 존재를 누적: 이미 있다면 count 증가, 없으면 1로 초기화
                event_patterns[best_pattern] = event_patterns.get(best_pattern, 0) + 1
                # 선택된 패턴의 길이만큼 토큰 건너뛰기 (non-overlapping 추출)
                i += len(best_pattern)
                
            all_results.append({"event_type": event_type, "time": event_time, "patterns": event_patterns})
            
        return all_results

    def extract_utterance_features(self) -> List[Dict]:
        """
        각 이벤트에 대한 화자별 발화 횟수를 추출합니다.
        
        Returns:
            list: 각 이벤트의 발화 특성 딕셔너리 리스트
        """
        utterance_features = []
        for event in self.event_data:
            seq = event.get("sequence", [])
            utterance_features.append({
                "total_utterance": len(seq),
                "utterance_counts": dict(Counter(seq))
            })
        return utterance_features

    def extract_features(self, CI: int) -> List[Dict]:
        """
        모든 특성을 추출하고 통합합니다.
        
        Args:
            CI: T-pattern 추출을 위한 Critical Interval (밀리초 단위)
            
        Returns:
            list: 각 이벤트에 대한 모든 특성이 포함된 리스트
        """
        t_patterns = self.extract_t_patterns(CI)
        utterance_features = self.extract_utterance_features()
        features = []
        
        for event, t_pat, utt_feat in zip(self.event_data, t_patterns, utterance_features):
            seq = event.get("sequence", [])
            
            # n-gram 추출
            ngram2, ngram3 = self.extract_ngrams(seq)
            
            # LSA Z-score 계산
            lsa_z = self.compute_lsa_z_scores(seq)
            
            # 순차 패턴은 n-gram과 동일하게 처리
            
            features.append({
                "event_type": event.get("event_type"),
                "time": event.get("time"),
                "gold": event.get("gold"),
                "success": event.get("success"),
                "total_utterance": len(seq),
                "entropy": self.compute_entropy(seq),
                "lsa_z": lsa_z,
                "ngram2": ngram2,
                "ngram3": ngram3,
                "t_patterns": t_pat.get("patterns", {}),
                "utterance_counts": utt_feat.get("utterance_counts"),
            })
            
        return features