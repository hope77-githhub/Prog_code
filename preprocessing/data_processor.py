"""
인게임 데이터 및 보정값 처리를 위한 IngameDataProcessor 클래스
"""

import json
from utils import convert_to_unix_timestamp_kst

class IngameDataProcessor:
    def __init__(self, ingame_json_path, voice_info_path):
        """
        IngameDataProcessor 클래스 초기화
        
        Args:
            ingame_json_path: 인게임 JSON 파일 경로
            voice_info_path: 음성 정보 파일 경로
        """
        self.ingame_json_path = ingame_json_path
        self.voice_info_path = voice_info_path
        self.full_data = None          # 전체 ingame JSON 데이터
        self.ingame_info = None        # info 섹션 (여기서 frames 포함)
        self.gameStartTimestamp = None # 게임 시작 타임스탬프 (밀리초)
        self.voice_start_time = None   # voice 녹음 시작 시간 (밀리초)
        self.offset = None

    def load_ingame_data(self):
        """
        인게임 JSON 데이터 로드
        
        Returns:
            dict: 로드된 인게임 데이터 또는 None (실패시)
        """
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
        인게임 데이터에서 가장 작은 realTimestamp 값을 게임 시작 타임스탬프로 사용
        
        Returns:
            int: 게임 시작 타임스탬프 (밀리초) 또는 None (찾지 못한 경우)
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
        """
        음성 정보 파일에서 녹음 시작 시간 로드
        
        Returns:
            int: 음성 녹음 시작 시간 (밀리초) 또는 None (찾지 못한 경우)
        """
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
        """
        인게임 데이터와 음성 데이터 간의 시간 오프셋 계산
        
        Returns:
            int: 시간 오프셋 (밀리초) 또는 None (필요 정보가 없는 경우)
        """
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