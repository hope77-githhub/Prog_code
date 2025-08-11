"""
유틸리티 함수 모음
"""

from datetime import datetime, timedelta
from typing import List, Dict

def convert_to_unix_timestamp_kst(timestamp):
    """UTC 타임스탬프를 KST로 변환하고 Unix 밀리초로 반환"""
    dt_utc = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    dt_kst = dt_utc + timedelta(hours=9)
    unix_timestamp = int(dt_kst.timestamp() * 1000)  # 밀리초 단위로 변환
    return unix_timestamp

# 전역 변수 (주의: 필요시 global 키워드 사용 필요)
gold_list = []

def gold_calculator(num, frame):
    """팀별 골드 계산 함수"""
    global gold_list
    blue_team = 0
    red_team = 0
    for i in range(1, 11):
        total_gold = frame[f'{i}']['totalGold']
        if i <= 5:
            blue_team += total_gold
        else:
            red_team += total_gold
    gold_list.append({"blue": blue_team, "red": red_team})

def reset_gold_list():
    """gold_list 변수 초기화"""
    global gold_list
    gold_list = []

def get_gold_list():
    """현재 gold_list 반환"""
    global gold_list
    return gold_list.copy()