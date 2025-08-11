"""
이벤트 기반 패턴 추출 및 분석 코드
"""

# ====================================================
# 모듈 임포트 정리
# ====================================================
import os
import json
import librosa
import numpy as np
import pandas as pd
import pprint
import sys
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import Counter
from itertools import product

# ====================================================
# 유틸리티 함수
# ====================================================
def convert_to_unix_timestamp_kst(timestamp):
    """UTC 타임스탬프를 KST로 변환하고 Unix 밀리초로 반환"""
    dt_utc = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    dt_kst = dt_utc + timedelta(hours=9)
    unix_timestamp = int(dt_kst.timestamp() * 1000)  # 밀리초 단위로 변환
    return unix_timestamp

gold_list = []

def gold_calculator(num, frame):
    """팀별 골드 계산 함수"""
    blue_team = 0
    red_team = 0
    for i in range(1, 11):
        total_gold = frame[f'{i}']['totalGold']
        if i <= 5:
            blue_team += total_gold
        else:
            red_team += total_gold
    gold_list.append({"blue": blue_team, "red": red_team})

# ====================================================
# 1. VoiceProcessor: 음성 데이터 처리 클래스
# ====================================================
class VoiceProcessor:
    def __init__(self, voice_root_directory, sr=22050, hop_length=512, csv_base_filename="voiced_flags"):
        self.voice_root_directory = voice_root_directory
        self.sr = sr
        self.hop_length = hop_length
        self.csv_base_filename = csv_base_filename
        self.voiced_flags = {}  # {파일명: voiced_flag (numpy array)}
        self.results = {}       # {파일명: utterance segmentation 결과 (리스트)}

    def get_audio_file_paths(self, extension=".wav"):
        file_paths = []
        for root, _, files in os.walk(self.voice_root_directory):
            for file in files:
                if file.endswith(extension):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def voiced_check(self, file_name, file_path):
        y, sr = librosa.load(file_path, sr=self.sr)
        print(f'{file_name} Sampling Rate is.. {sr}')
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, sr=sr,
            fmin=librosa.note_to_hz('C2'), #E2
            fmax=librosa.note_to_hz('C5') #C5
        )
        return voiced_flag

    def process_file(self, path):
        file_name = os.path.splitext(os.path.basename(path))[0]
        voiced_flag = self.voiced_check(file_name, path)
        return file_name, voiced_flag

    def process_files_parallel(self, num_workers=None):
        file_paths = self.get_audio_file_paths()
        results = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for file_name, voiced_flag in tqdm(executor.map(self.process_file, file_paths), total=len(file_paths)):
                results[file_name] = voiced_flag
        self.voiced_flags = results
        print("\nVoice flags 데이터:")
        for key, value in results.items():
            data_shape = value.shape if hasattr(value, 'shape') else 'N/A'
            print(f"{key}: type={type(value)}, shape={data_shape}")
        return results

    def save_voiced_flags_csv(self, starting_number=1):
        csv_filename = f"{self.csv_base_filename}_{starting_number}"
        while os.path.exists(csv_filename):
            starting_number += 1
            csv_filename = f"{self.csv_base_filename}_{starting_number}"
        data = []
        for file_name, voiced_flag in self.voiced_flags.items():
            data.append({
                "file_name": file_name,
                "voiced_flag": json.dumps(voiced_flag.tolist())
            })
        df = pd.DataFrame(data)
        df.to_csv(csv_filename, index=False)
        print(f"CSV 파일이 {csv_filename}로 저장되었습니다.")
        return csv_filename

    def load_voiced_flags(self, csv_filename):
        df = pd.read_csv(csv_filename)
        voiced_flags = {}
        for index, row in df.iterrows():
            file_name = row["file_name"]
            voiced_flag_list = json.loads(row["voiced_flag"])
            voiced_flag_array = np.array(voiced_flag_list)
            # print(len(voiced_flag_array))배열의 길이 확인 
            voiced_flags[file_name] = voiced_flag_array
        self.voiced_flags = voiced_flags
        return voiced_flags

    #voiced_flag Threshold 
    def utterance_segmentation(self, voice_data, threshold=100): 
        sorted_utterance = []
        cur_t = 0
        cur_f = 0
        sf = None
        ef = None
        for i in range(len(voice_data)):
            if voice_data[i]:  # True일 때
                if cur_t == 0:
                    sf = i
                # 다음 프레임이 False이면 잠재적인 종료 지점으로 저장
                if i < len(voice_data) - 1 and not voice_data[i + 1]:
                    ef = i
                cur_t += 1
                cur_f = 0
            else:
                cur_f += 1
                # 무음이 threshold 이상 연속되고, 음성 구간이 존재하면 utterance 종료
                if cur_f > threshold and cur_t > 0:
                    if ef is None:
                        ef = i - cur_f  # 마지막 True 프레임 인덱스 추정
                    if ef - sf > 10:  # 최소 구간 길이 조건
                        sorted_utterance.append([sf, ef, ef - sf])
                    cur_t = 0
                    cur_f = 0
                    sf = None
                    ef = None
        # 반복문 종료 후, 아직 완료되지 않은 음성 구간 처리
        if cur_t > 0 and sf is not None:
            if ef is None:
                ef = len(voice_data) - 1
            if ef - sf > 10:
                sorted_utterance.append([sf, ef, ef - sf])
        return sorted_utterance

    def frame_to_time(self, sorted_range):
        sorted_time = []
        for r in sorted_range:
            # ms 단위로 변환하여 ms단위로 모든 데이터를 통일. (1000ms = 1s)
            start_time = round(librosa.frames_to_time(r[0], sr=self.sr, hop_length=self.hop_length) * 1000)
            end_time = round(librosa.frames_to_time(r[1], sr=self.sr, hop_length=self.hop_length) * 1000) 
            sorted_time.append({"start": start_time, "end": end_time})
        return sorted_time

    def process_utterances(self, threshold=100):
        results = {}
        for file_name, flag in self.voiced_flags.items():
            sorted_utterance = self.utterance_segmentation(flag, threshold=threshold)
            sorted_time = self.frame_to_time(sorted_utterance)
            results[file_name] = sorted_time
        self.results = results
        return results

# ====================================================
# 2. IngameDataProcessor: 인게임 데이터 및 보정값 처리 클래스
# ====================================================
class IngameDataProcessor:
    def __init__(self, ingame_json_path, voice_info_path):
        self.ingame_json_path = ingame_json_path
        self.voice_info_path = voice_info_path
        self.full_data = None          # 전체 ingame JSON 데이터
        self.ingame_info = None        # info 섹션 (여기서 frames 포함)
        self.gameStartTimestamp = None # 게임 시작 타임스탬프 (밀리초)
        self.voice_start_time = None   # voice 녹음 시작 시간 (밀리초)
        self.offset = None

    def load_ingame_data(self):
        try:
            with open(self.ingame_json_path, 'r') as json_file:
                data = json.load(json_file)
            self.full_data = data
            self.ingame_info = data.get('info', None)
            # info 내의 frames → events에서 게임 시작 타임스탬프를 추출
            self.gameStartTimestamp = self.get_game_start_timestamp()
            if self.gameStartTimestamp is None:
                print("Warning: 'realTimestamp'를 찾을 수 없습니다.")
            return data
        except FileNotFoundError:
            print(f"Ingame JSON 파일을 찾을 수 없습니다: {self.ingame_json_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return None

    def get_game_start_timestamp(self):
        """
        ingame_info 내의 frames에서 각 프레임의 events를 순회하여 가장 작은 realTimestamp 값을
        게임 시작 타임스탬프로 사용합니다.
        """
        if not self.ingame_info or "frames" not in self.ingame_info:
            return None
        min_ts = None
        for frame in self.ingame_info["frames"]:
            if "events" in frame:
                for event in frame["events"]:
                    if "realTimestamp" in event:
                        ts = event["realTimestamp"]
                        if min_ts is None or ts < min_ts:
                            min_ts = ts
        return min_ts

    def load_voice_start_time(self):
        try:
            with open(self.voice_info_path, 'r') as f:
                for line in f:
                    if "Start time" in line:
                        _, _, timestamp_str = line.partition(":")
                        timestamp_str = timestamp_str.strip()
                        # convert_to_unix_timestamp_kst 함수를 사용하여 Unix timestamp (밀리초 단위)로 변환
                        self.voice_start_time = convert_to_unix_timestamp_kst(timestamp_str)
                        return self.voice_start_time
            print("Warning: 'Start time' 항목을 찾지 못했습니다.")
            return None
        except FileNotFoundError:
            print(f"Voice info 파일을 찾을 수 없습니다: {self.voice_info_path}")
            return None

    def compute_offset(self):
        if self.gameStartTimestamp is None:
            print("Error: ingame 게임 시작 타임스탬프가 없습니다.")
            return None
        if self.voice_start_time is None:
            print("Error: voice recording 시작 시간이 없습니다.")
            return None
        
        # 모든 단위는 ms단위로 통일 
        self.offset = abs(self.voice_start_time - self.gameStartTimestamp)
        print(f"계산된 offset (밀리초): {self.offset}")
        return self.offset

# ====================================================
# 3. EventClassifier: 인게임 이벤트 분류 클래스
# ====================================================
class EventClassifier:
    def __init__(self, loaded_data, team_color):
        """
        :param loaded_data: ingame 데이터의 JSON 객체 (특히 'frames' 포함)
        :param team_color: 팀 번호 (1: Blue, 2: Red)
        """
        self.loaded_data = loaded_data
        self.team_color = team_color
        self.event_data = {
            'CHAMPION_KILL': [],
            'ELITE_MONSTER_KILL': [],
            'BUILDING_KILL': [],
            'TURRET_PLATE_DESTROYED': []
        }
        self.assist_gold_mapping = {
            400: 150,
            300: 150,
            267: 134,
            200: 100,
            150: 75,
            112: 56,
            100: 50
        }

    def handle_champion_kill(self, event):
        if 'assistingParticipantIds' in event:
            killer_id = event.get('killerId', 0)
            bounty = event.get('bounty', 0)
            success = 1 if ((self.team_color == 1 and killer_id < 6) or (self.team_color == 2 and killer_id >= 6)) else 0
            assist_gold = self.assist_gold_mapping.get(bounty, 0)
            total_gold = bounty + assist_gold
            self.event_data['CHAMPION_KILL'].append({
                'time': event['timestamp'],
                'gold': total_gold,
                'success': success
            })

    def handle_elite_monster_kill(self, event):
        if 'assistingParticipantIds' in event:
            killer_id = event.get('killerId', 0)
            success = 1 if ((self.team_color == 1 and killer_id < 6) or (self.team_color == 2 and killer_id >= 6)) else 0
            monster_type = event.get('monsterType', '')
            dragon_type = event.get('monsterSubType', '')
            gold = 0
            if monster_type == 'DRAGON' and monster_type != 'ELDER_DRAGON':
                if dragon_type == 'FIRE_DRAGON':
                    gold = 522
                elif dragon_type == 'WATER_DRAGON':
                    gold = 500
                elif dragon_type == 'EARTH_DRAGON':
                    gold = 843
                elif dragon_type == 'AIR_DRAGON':
                    gold = 1008
                elif dragon_type == 'CHEMTECH_DRAGON':
                    gold = 2500
                elif dragon_type == 'HEXTECH_DRAGON':
                    gold = 2425
            elif monster_type == 'ELDER_DRAGON':
                gold = 1250
            elif monster_type == 'HORDE':
                gold = 60
            elif monster_type == 'BARON_NASHOR':
                gold = 1525
            elif monster_type == 'RIFTHERALD':
                gold = 200
            self.event_data['ELITE_MONSTER_KILL'].append({
                'time': event['timestamp'],
                'gold': gold,
                'success': success
            })

    def handle_building_kill(self, event):
        killer_id = event.get('killerId', 0)
        building_type = event.get('buildingType', '')
        tower_type = event.get('towerType', '')
        success = 1 if ((self.team_color == 1 and killer_id < 6) or (self.team_color == 2 and killer_id >= 6)) else 0
        gold = 0
        if building_type == 'TOWER_BUILDING':
            if tower_type == 'OUTER_TURRET':
                gold = 425
            elif tower_type == 'INNER_TURRET':
                gold = 775
            elif tower_type == 'BASE_TURRET':
                gold = 475
            elif tower_type == 'NEXUS_TURRET':
                gold = 250
        elif building_type == 'INHIBITOR_BUILDING':
            gold = 175
        self.event_data['BUILDING_KILL'].append({
            'time': event['timestamp'],
            'gold': gold,
            'success': success
        })

    def handle_turret_plate_destroyed(self, event):
        killer_id = event.get('killerId', 0)
        success = 1 if ((self.team_color == 1 and killer_id < 6) or (self.team_color == 2 and killer_id >= 6)) else 0
        gold = 125
        self.event_data['TURRET_PLATE_DESTROYED'].append({
            'time': event['timestamp'],
            'gold': gold,
            'success': success
        })

    def classify(self):
        gold_list = []  # gold_calculator 함수에서 사용할 리스트 정의
        data = self.loaded_data['info']["frames"]

        for i, frame in enumerate(data):
            # 골드 계산 
            gold_calculator(i, frame['participantFrames'])
            for event in frame['events']:
                event_type = event['type']

                if event_type == 'CHAMPION_KILL':
                    self.handle_champion_kill(event)
                elif event_type == 'ELITE_MONSTER_KILL':
                    self.handle_elite_monster_kill(event)
                elif event_type == 'BUILDING_KILL':
                    self.handle_building_kill(event)
                elif event_type == 'TURRET_PLATE_DESTROYED':
                    self.handle_turret_plate_destroyed(event)

        return self.event_data

# ====================================================
# 4. SequenceGenerator: 이벤트 시퀀스 생성 클래스
# ====================================================
class SequenceGenerator:
    def __init__(self, voice_results, offset, team_color):
        """
        :param voice_results: VoiceProcessor에서 생성한 utterance segmentation 결과 
                              (예: {'top1': [...], 'jug4': [...], ...})
        :param offset: 인게임 데이터와 음성 데이터 간 시간 차 보정값 (초 단위)
        :param team_color: 팀 번호 (1: Blue, 2: Red)
        """
        self.voice_results = voice_results
        self.offset = offset
        self.team_color = team_color

    # ===========================
    # event window sliding 
    # ===========================
    def extract_window_sliding(self, event_data, window_size: int, step_size: int):
        """
        이벤트 기준으로 슬라이싱된 데이터를 기반으로, 지정한 윈도우 크기(window_size)와 슬라이딩 간격(step_size)을 사용하여
        데이터 증강을 진행합니다.
        
        :param event_data: 이벤트 기준 데이터 (기존과 동일 형식)
        :param window_size: 윈도우의 길이 (밀리초 단위)
        :param step_size: 슬라이딩 간격 (밀리초 단위)
        :return: 증강된 이벤트 시퀀스 리스트
        """
        base_keys = ['top', 'jug', 'mid', 'adc', 'sup']
        line_abbr = ['t', 'j', 'm', 'a', 's']
        augmented_event_sequences = []
        
        for event_type, events in event_data.items():
            for event in events:
                # event_time에 offset을 더해 voice_data와 timeline을 맞추기, 60초의 구간을 더하고 빼기. 
                event_time = event['time']
                overall_start = event_time - 60000  # 이벤트 발생 60초 전
                overall_end = event_time + 60000    # 이벤트 발생 60초 후
                
                # np.arange로 윈도우 시작점을 생성 (동일한 step_size 간격)
                window_starts = np.arange(overall_start, overall_end - window_size + step_size, step_size)
                
                for win_s in window_starts:
                    win_e = win_s + window_size
                   
                    sequence_data = []
                    for base_key, abbr in zip(base_keys, line_abbr):
                        for key in self.voice_results:
                            if key.startswith(base_key):
                                data = self.voice_results[key]
                                for time_obj in data:
                                    voice_start = time_obj['start'] - self.offset
                                    voice_end = time_obj['end'] - self.offset
                                    # 해당 윈도우 내에 시작 시간이 포함되면 데이터 포함
                                    if win_s <= voice_start <= win_e:
                                        sequence_data.append({
                                            'line': abbr,
                                            'start': voice_start,
                                            'end': voice_end
                                        })

                    # 발화 데이터가 없으면 건너뛰기
                    if not sequence_data:
                        continue

                    # 시작 시간 기준으로 정렬
                    sequence_data.sort(key=lambda x: x['start'])
                    sequence_abbr = [item['line'] for item in sequence_data]
                    sequence_start = [item['start'] for item in sequence_data]
                    sequence_end = [item['end'] for item in sequence_data]

                    # 윈도우의 중심 시간을 계산하여 time으로 기록
                    augmented_event_sequences.append({
                        'event_type': event_type,
                        'success': event.get('success', 0),
                        'gold': event.get('gold', 0),
                        'time': event_time,
                        'sequence': sequence_abbr,
                        'start_time': sequence_start,
                        'end_time': sequence_end
                    })
                    
        return augmented_event_sequences

# ====================================================
# 5. Feature_Extractor_event: 이벤트 특성 추출 클래스
# ====================================================
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
        self.speakers = ["t", "j", "m", "a", "s"]  # 기준 발화자 순서 저장

    def compute_entropy(self, seq: List[str]) -> float:
        """주어진 sequence의 엔트로피 계산"""
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
        """길이 2 전이(2-gram)에 대한 Z-score를 계산합니다."""
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
        
        Returns:
            Tuple[Dict, Dict]: (2-gram 딕셔너리, 3-gram 딕셔너리)
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
        filtered = []
        for (s_idx, e_idx) in indices:
            if s_idx < e_idx < len(start_times):
                is_valid = True
                for k in range(s_idx, e_idx):
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
        만약 3-gram 패턴이 포착되면 그 패턴의 시작 이후부터 다시 탐색하며,
        동일 패턴이 CI 조건을 만족하면 기존 카운트에 누적합니다.
        
        Returns:
            List of dict: 각 이벤트의 event_type, time 및 t-pattern dictionary.
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
        
        Parameters:
            CI: T-pattern 추출을 위한 Critical Interval (밀리초 단위)
            
        Returns:
            List[Dict]: 각 이벤트에 대한 모든 특성이 포함된 리스트
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

# ====================================================
# 6. EventDatasetBuilder: 데이터셋 구축 클래스
# ====================================================
class EventDatasetBuilder:
    def __init__(self, csv_file: str, game_num: int):
        """
        :param csv_file: 이벤트 데이터셋을 저장할 CSV 파일 경로.
        :param game_num: 이 클래스 인스턴스가 담당할 게임(매치) 번호
        """
        self.csv_file = csv_file
        self.game_num = game_num
        
        # 기준 발화자 순서
        self.speakers = ["t", "j", "m", "a", "s"]
        
        # Base columns: 이벤트 정보
        self.base_columns = ["game_num", "event_label", "time", "gold", "success", "total_utterance", "entropy"]
        
        # LSA Z-score 컬럼 추가: 모든 가능한 2개 화자 조합 (5x5=25개)
        self.lsa_columns = [f"lsa_z_{t1}{t2}" for t1 in self.speakers for t2 in self.speakers]

        
        # t_patterns: 유의한 패턴 
        # 2-gram: 발화자가 다름
        self.tpattern2 = [f"tpattern_{'_'.join(p)}" for p in product(self.speakers, repeat=2) if p[0] != p[1]]

        # 3-gram: 연속된 동일 발화자가 없는 패턴 (즉, p[0]!=p[1] 그리고 p[1]!=p[2])
        self.tpattern3 = [f"tpattern_{'_'.join(p)}" for p in product(self.speakers, repeat=3) 
                        if p[0] != p[1] and p[1] != p[2]]

        self.tpattern_columns = self.tpattern2 + self.tpattern3

        # utterance_counts: 각 화자별 utterance 카운트
        self.utterance_columns = [f"utterance_{s}" for s in self.speakers]

        # 최종적으로 사용할 전체 컬럼 리스트
        self.all_columns = (
            self.base_columns +
            self.lsa_columns +
            self.tpattern_columns + 
            self.utterance_columns
        )
        
        # 이벤트 라벨 매핑 딕셔너리
        self.event_label_map = {
            "CHAMPION_KILL": 1,
            "ELITE_MONSTER_KILL": 2,
            "TURRET_PLATE_DESTROYED": 3,
            "BUILDING_KILL": 4
        }
        
        # csv_file이 이미 존재하면 읽어서 DataFrame으로 로드, 아니면 빈 DataFrame 생성
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
        else:
            self.df = pd.DataFrame(columns=self.all_columns)
    
    def add_features(self, event_features: list):
        """
        이벤트별 feature 딕셔너리 리스트를 받아 CSV 파일에 추가 저장합니다.
        
        Parameters:
            event_features (List[Dict]): Feature_Extractor_event.extract_features()의 결과 리스트
        """
        rows = []
        
        for feat in event_features:
            row = {}
            # 현재 클래스 인스턴스에 설정된 게임 번호를 사용
            row["game_num"] = self.game_num
            
            # event_label: event_type을 숫자로 변환
            event_type = feat.get("event_type")
            row["event_label"] = self.event_label_map.get(event_type, 0)  # 미정의 event_type은 0
            
            # 기본 정보
            row["time"] = feat.get("time")
            row["gold"] = feat.get("gold")
            row["success"] = feat.get("success")
            row["total_utterance"] = feat.get("total_utterance")
            row["entropy"] = feat.get("entropy")
            
            # LSA Z-score 추가
            lsa_z_dict = feat.get("lsa_z", {})
            for t1 in self.speakers:
                for t2 in self.speakers:
                    key = f"{t1}{t2}"
                    col_name = f"lsa_z_{key}"
                    row[col_name] = lsa_z_dict.get(key, 0.0)
            
            # t_patterns
            tpattern_dict = feat.get("t_patterns", {})
            for p2 in product(self.speakers, repeat=2):
                if p2[0] != p2[1]:  # 다른 화자끼리만
                    col_name = f"tpattern_{'_'.join(p2)}"
                    row[col_name] = tpattern_dict.get(p2, 0)
            
            for p3 in product(self.speakers, repeat=3):
                if len(set(p3)) == 3:  # 모두 다른 화자
                    col_name = f"tpattern_{'_'.join(p3)}"
                    row[col_name] = tpattern_dict.get(p3, 0)
            
            # utterance_counts
            utterance_dict = feat.get("utterance_counts", {})
            for s in self.speakers:
                col_name = f"utterance_{s}"
                row[col_name] = utterance_dict.get(s, 0)
            
            rows.append(row)
        
        # 빈 데이터가 없는 경우에만 DataFrame 생성 및 저장
        if rows:
            new_df = pd.DataFrame(rows)
            
            # 컬럼 순서 조정
            for col in self.all_columns:
                if col not in new_df.columns:
                    new_df[col] = 0
            
            new_df = new_df[self.all_columns]
            
            # CSV 파일 존재 시 기존 데이터와 병합
            if os.path.exists(self.csv_file):
                existing_df = pd.read_csv(self.csv_file)
                self.df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                self.df = new_df.copy()
            
            self.df.to_csv(self.csv_file, index=False)
            print(f"Added {len(rows)} new rows to dataset")
        else:
            print("No valid data to add to dataset")
    
    def inspect_dataset(self):
        """
        CSV 파일에 저장된 데이터셋 정보를 확인합니다.
        """
        if os.path.exists(self.csv_file):
            df = pd.read_csv(self.csv_file)
            print("Dataset summary:")
            print(f"- Total rows: {len(df)}")
            print(f"- Event distribution: {df['event_label'].value_counts().to_dict()}")
            print(f"- Games included: {sorted(df['game_num'].unique())}")
            print(f"- Total columns: {len(df.columns)}")
        else:
            print(f"CSV file '{self.csv_file}' does not exist.")

# ====================================================
# Main 실행: 각 클래스 인스턴스를 생성하고 유기적으로 연결
# ====================================================
if __name__ == "__main__":
    # 전체 데이터셋을 저장할 CSV 파일 경로
    dataset_csv = "lol_event_dataset.csv"
    
    # 게임 번호 루프
    for game_num in range(1, 5):
        print(f"\n{'='*50}")
        print(f"Processing game {game_num}...")
        print(f"{'='*50}")
        
        # 파일 경로 설정
        voice_root_directory = f"/Users/jh/PycharmProjects/Research/Data/voice_data/game{game_num}"
        voice_info_path = f"/Users/jh/PycharmProjects/Research/Data/voice_data/game{game_num}/info_{game_num}.txt"
        ingame_json_path = f"/Users/jh/PycharmProjects/Research/Data/ingame_data/match{game_num}.json"
    
        # 1. 음성 데이터 처리
        csv_filename = f"voiced_flags_{game_num}.csv"
        voice_processor = VoiceProcessor(voice_root_directory, sr=22050, hop_length=512, csv_base_filename=csv_filename)
        if os.path.exists(csv_filename):
            print(f"Loading existing voice flags from {csv_filename}")
            voice_processor.load_voiced_flags(csv_filename)
        else:
            print("Processing audio files to extract voice flags")
            voice_processor.process_files_parallel()
            csv_filename = voice_processor.save_voiced_flags_csv()
        
        print("Segmenting utterances")
        voice_results = voice_processor.process_utterances()
        break

        # 2. 인게임 데이터 처리
        print("\nProcessing ingame data")
        ingame_processor = IngameDataProcessor(ingame_json_path, voice_info_path)
        ingame_processor.load_ingame_data()
        ingame_processor.load_voice_start_time()
        offset = ingame_processor.compute_offset()
        
        if offset is None:
            print(f"Error: Could not compute offset for game {game_num}. Skipping...")
            continue
    
        # 3. 이벤트 분류
        print("\nClassifying game events")
        # 팀 색상 설정 (자동화를 위해 팀 색상 결정 로직을 수정)
        team_color = 1  # 블루 팀 기본값
        if game_num == 2 or game_num == 4:
            team_color = 2  # 레드 팀 (게임 2, 4는 레드 팀으로 설정)
        
        # 인게임 이벤트 분류
        gold_list = []  # gold_calculator 함수에서 사용하는 리스트 초기화
        event_classifier = EventClassifier(ingame_processor.full_data, team_color)
        event_data = event_classifier.classify()     
        
        # 4. 이벤트 시퀀스 추출 
        print("\nGenerating event sequences")
        sequence_generator = SequenceGenerator(voice_results, offset, team_color)
        event_sequences = sequence_generator.extract_window_sliding(
            event_data, window_size=30000, step_size=1000
        )
        print(f"Generated {len(event_sequences)} event sequences")
        
        # 5. 패턴 추출 및 특성 추출
        print("\nExtracting features")
        CI = 1000  # Critical Interval (밀리초 단위)
        event_feature_extractor = Feature_Extractor_event(event_sequences)
        event_features = event_feature_extractor.extract_features(CI)
        print(f"Extracted features from {(event_features)} event sequences")
        
        
        # 6. 데이터셋 구축
        print("\nBuilding dataset")
        event_builder = EventDatasetBuilder(dataset_csv, game_num)
        event_builder.add_features(event_features)
        
        print(f"Completed processing for game {game_num}")
    
        # 모든 게임 처리 후 최종 데이터셋 검사
    print("\n\nFinal dataset inspection:")
    event_builder = EventDatasetBuilder(dataset_csv, 0)  # game_num은 중요하지 않음
    event_builder.inspect_dataset()
        
    print("\nProcessing complete!")
