"""
설정 파일 - 하이퍼파라미터 및 경로 설정
"""

# 파일 경로 설정
DATASET_CSV = "lol_event_dataset.csv"

# 음성 처리 관련 설정
VOICE_PARAMS = {
    "sr": 22050,
    "hop_length": 512
}

# 이벤트 시퀀스 생성 설정
SEQUENCE_PARAMS = {
    "window_size": 30000,  # 밀리초 단위 (30초)
    "step_size": 1000      # 밀리초 단위 (1초)
}

# 특성 추출 관련 설정
FEATURE_PARAMS = {
    "CI": 5000  # Critical Interval (밀리초 단위)
}

# 게임별 팀 색상 설정 (1: Blue, 2: Red)
TEAM_COLORS = {
    1: 1,  # 게임 1: Blue 팀
    2: 2,  # 게임 2: Red 팀
    3: 1,  # 게임 3: Blue 팀
    4: 2   # 게임 4: Red 팀
}

# 게임별 파일 경로 포맷 설정
PATH_FORMAT = {
    "voice_root": "/Users/jh/PycharmProjects/Research/Data/voice_data/game{game_num}",
    "voice_info": "/Users/jh/PycharmProjects/Research/Data/voice_data/game{game_num}/info_{game_num}.txt",
    "ingame_json": "/Users/jh/PycharmProjects/Research/Data/ingame_data/match{game_num}.json",
    "csv_filename": "voiced_flags_{game_num}.csv"
}

# 이벤트 라벨 매핑
EVENT_LABEL_MAP = {
    "CHAMPION_KILL": 1,
    "ELITE_MONSTER_KILL": 2,
    "TURRET_PLATE_DESTROYED": 3,
    "BUILDING_KILL": 4
}

# 발화자 목록
SPEAKERS = ["t", "j", "m", "a", "s"]  # top, jungle, mid, adc, support

# 보조금 매핑 (Champion Kill 시 assist gold)
ASSIST_GOLD_MAPPING = {
    400: 150,
    300: 150,
    267: 134,
    200: 100,
    150: 75,
    112: 56,
    100: 50
}