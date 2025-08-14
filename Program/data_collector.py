import os
import sys
import requests
import logging
import math
import threading
import time
import zipfile
import smtplib
import json
import ctypes
from email.mime.multipart import MIMEMultipart 
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email.header import Header
from email import encoders
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# 인게임 데이터 수집 프로그램, 데모버전 

# 로그 설정
LOG_FILE = os.path.join(os.path.expanduser("~"), "lol_data_collector.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# 설정 파일 경로
CONFIG_FILE = os.path.join(os.path.expanduser("~"), "lol_data_collector_config.json")

# 전역 변수로 팀명과 라벨 설정
TEAM_NAME = ""
TEAM_LABEL = ""
LABEL_DESCRIPTIONS = {
    1: "1군 LCK",
    2: "2군 CL",
    3: "3군 아카데미",
    4: "아카데미"
}

def is_admin():
    """관리자 권한으로 실행 중인지 확인"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def load_config():
    """설정 파일에서 팀명과 라벨 로드"""
    global TEAM_NAME, TEAM_LABEL
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                TEAM_NAME = config.get("team_name", "")
                TEAM_LABEL = config.get("team_label", "")
                logging.info(f"설정 파일에서 로드: 팀명={TEAM_NAME}, 라벨={TEAM_LABEL}")
                return True
        return False
    except Exception as e:
        logging.error(f"설정 파일 로드 실패: {e}")
        return False

def save_config():
    """설정 파일에 팀명과 라벨 저장"""
    try:
        config = {
            "team_name": TEAM_NAME,
            "team_label": TEAM_LABEL
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False)
        logging.info(f"설정 파일 저장 완료: {CONFIG_FILE}")
        return True
    except Exception as e:
        logging.error(f"설정 파일 저장 실패: {e}")
        return False

def normalize_summoner_name(name: str) -> str:
    """소환사명을 정규화합니다."""
    return name.split('#')[0].strip()

def get_team_info():
    """프로그램 시작 시 팀명과 라벨을 입력받습니다."""
    global TEAM_NAME, TEAM_LABEL
    
    if load_config():
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "--background":
        TEAM_NAME = "DefaultTeam"
        TEAM_LABEL = "1"
        logging.info(f"백그라운드 모드에서 기본값 사용: 팀명={TEAM_NAME}, 라벨={TEAM_LABEL}")
        save_config()
        return
    
    print("리그 오브 레전드 게임 데이터 수집 프로그램을 시작합니다.")
    TEAM_NAME = input("팀명을 입력해주세요: ")
    
    print("\n라벨을 선택해주세요:")
    for num, desc in LABEL_DESCRIPTIONS.items():
        print(f"{num}: {desc}")
    
    while True:
        try:
            label_num = int(input("번호를 입력하세요 (1-4): "))
            if label_num in LABEL_DESCRIPTIONS:
                TEAM_LABEL = str(label_num)
                break
            else:
                print("1에서 4 사이의 숫자를 입력해주세요.")
        except ValueError:
            print("유효한 숫자를 입력해주세요.")
    
    print(f"\n팀명: {TEAM_NAME}, 라벨: {TEAM_LABEL} ({LABEL_DESCRIPTIONS[int(TEAM_LABEL)]})")
    print("데이터 수집을 시작합니다. 게임이 시작될 때까지 대기합니다...\n")
    save_config()

def compress_and_send_files(folder_path):
    """Compress the game data folder and send it via email."""
    try:
        zip_path = f"{folder_path}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)
        
        # --- 여기부터 수정된 부분 ---

        # email configuration 
        sender_email = "datacollection.ixlab@gmail.com"
        sender_password = ""
        receiver_email = "datacollection.ixlab@gmail.com"

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Date'] = formatdate(localtime=True)
        
        # 1. 제목(Subject)에 한글이 포함될 경우를 대비해 UTF-8로 인코딩
        subject_text = f"Game Data Collection - {os.path.basename(folder_path)}"
        msg['Subject'] = Header(subject_text, 'utf-8')

        # 2. 본문(Body)도 UTF-8로 인코딩
        body_text = f"Attached is the compressed game data from: {os.path.basename(folder_path)}"
        msg.attach(MIMEText(body_text, 'plain', 'utf-8'))
        
        with open(zip_path, 'rb') as f:
            part = MIMEBase('application', 'zip')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            
            # 3. 첨부 파일명(filename)도 UTF-8로 인코딩
            part.add_header('Content-Disposition', 'attachment', filename=str(Header(os.path.basename(zip_path), 'utf-8')))
            msg.attach(part)
            
        # --- 여기까지 수정 ---
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logging.info(f"Successfully sent compressed data to {receiver_email}")
        os.remove(zip_path)
    except Exception as e:
        logging.error(f"Failed to compress and send files: {e}")


class GameDataFetcher:
    """리그 오브 레전드 클라이언트 API를 통한 게임 데이터 수집 클래스"""
    
    def __init__(self, api_url="https://127.0.0.1:2999/liveclientdata/allgamedata", retries=3, backoff_factor=0.5):
        self.api_url = api_url
        self.retries = retries
        self.backoff_factor = backoff_factor
        # SSL 경고 비활성화 (로컬 API이므로 안전)
        requests.packages.urllib3.disable_warnings()
        self.lock = threading.Lock()

    def get_all_game_data(self):
        """재시도 로직을 포함한 리그 오브 레전드 클라이언트 API에서 모든 게임 데이터 가져오기"""
        for attempt in range(self.retries):
            try:
                response = requests.get(self.api_url, verify=False, timeout=5)
                response.raise_for_status()
                
                # JSON 응답인지 확인
                if response.headers.get("Content-Type", "").startswith("application/json"):
                    return response.json()
                
                logging.error("예상치 못한 응답 형식: %s", response.text)
                return None
                
            except requests.exceptions.HTTPError as e:
                if hasattr(e, 'response') and e.response.status_code == 404:
                    logging.debug("게임이 감지되지 않음. 게임 시작을 기다리는 중...")
                    return None
                else:
                    logging.error("HTTP 오류 발생: %s", e)
                    
            except requests.exceptions.ConnectionError:
                logging.debug("API에 연결할 수 없음. 가능한 원인: 게임이 실행되지 않음 또는 API에 접근할 수 없음")
                return None
                
            except requests.exceptions.Timeout:
                logging.error("타임아웃 오류: %s에 대한 요청이 시간 초과됨", self.api_url)
                
            except requests.RequestException as e:
                logging.error("%s 접근 중 오류: %s", self.api_url, e)
            
            # 재시도 전 백오프 대기
            if attempt < self.retries - 1:  # 마지막 시도가 아닌 경우에만 대기
                wait_time = self.backoff_factor * (2 ** attempt)
                logging.debug(f"재시도 전 {wait_time}초 대기 중...")
                time.sleep(wait_time)
        
        logging.debug("%d번 시도 후 게임 데이터 가져오기 실패", self.retries)
        return None

    def get_all_game_data_with_timing(self):
        """게임 데이터 조회 및 응답 시점 기록"""
        for attempt in range(self.retries):
            try:
                request_time_ms = time.time_ns() // 1000000
                response = requests.get(self.api_url, verify=False, timeout=5) 
                response.raise_for_status()
                
                if response.headers.get("Content-Type", "").startswith("application/json"):
                    game_data = response.json()
                    
                    # 게임 시간 추출 (초를 밀리초로 변환)
                    game_time_ms = int(game_data.get("gameData", {}).get("gameTime", 0.0) * 1000)
                    
                    # 시간 차이 계산 (시스템 시간 - 게임 시간)
                    time_difference_ms = request_time_ms - game_time_ms
                    
                    return game_data, request_time_ms, game_time_ms, time_difference_ms
                
                return None, None, None, None
                
            except requests.exceptions.HTTPError as e:
                if hasattr(e, 'response') and e.response.status_code == 404:
                    return None, None, None, None
                else:
                    logging.error("HTTP 오류 발생: %s", e)
            except requests.exceptions.ConnectionError:
                logging.debug("네트워크 연결 문제. 재시도...")
            except requests.exceptions.Timeout:
                logging.error("타임아웃 발생. 재시도...")
            
            if attempt < self.retries - 1:
                time.sleep(self.backoff_factor * (2 ** attempt))
                
        return None, None, None, None

    def wait_for_first_response(self):
        """첫 번째 성공적인 API 응답이 올 때까지 대기하고 타이밍 정보 수집"""
        logging.info("첫 번째 API 응답 대기 중... (1초 간격)")
        
        loop_count = 0
        start_time = time.time_ns() // 1_000_000  # 데이터 로깅 시작한 시간 

        while True:
            game_data, request_time_ms, game_time_ms, time_difference_ms = self.get_all_game_data_with_timing()
            
            if game_data is not None:
                logging.info("첫 번째 200 응답 수신 성공!")
                logging.info(f"게임시작시간: {time_difference_ms}ms")
                
                return game_data, request_time_ms, game_time_ms, time_difference_ms, start_time 
            
            loop_count += 1
            
            # 2초 대기 (서버 부하 감소)
            time.sleep(1.0)
            
            # 안전장치 (최대 20분)
            if loop_count > 600:  # 20분
                elapsed_time = (time.time_ns() // 1_000_000 - start_time) / 1000
                logging.warning(f"첫 번째 응답 대기 시간 초과 (20분) - 총 소요: {elapsed_time:.1f}초")
                break
                
        return None, None, None, None, None

class DataStorage:
    def __init__(self, game_folder):
        self.game_folder = Path(game_folder)
        self.game_folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.player_mapping = {}

    def save_data_to_csv(self, all_players_data, events_data):
        """이벤트 데이터와 플레이어 매핑 정보를 CSV 파일로 저장합니다."""
        self.update_player_mapping(all_players_data)
        processed_events_data = self.process_events_with_player_ids(events_data)
        events_df = pd.DataFrame(processed_events_data)
        info_data = []
        for summoner_name, data in self.player_mapping.items():
            info_data.append({
                "Team": data["Team"],
                "PlayerID": data["ID"],
                "Summoner Name": summoner_name,
                "Position": data["Position"]
            })
        info_df = pd.DataFrame(info_data)
        with self.lock:
            self._overwrite_csv(events_df, "event.csv")
            self._overwrite_csv(info_df, "info.csv")
            
    def update_player_mapping(self, all_players_data):
        """매치 종료 시, 최종 allPlayers 데이터를 활용하여 플레이어 매핑 정보를 생성합니다."""
        positions = {
            "ORDER": {"TOP": None, "JUNGLE": None, "MIDDLE": None, "BOTTOM": None, "UTILITY": None, "None": []},
            "CHAOS": {"TOP": None, "JUNGLE": None, "MIDDLE": None, "BOTTOM": None, "UTILITY": None, "None": []}
        }
        for player in all_players_data:
            team = player.get("Team", "Unknown")
            if team not in ["ORDER", "CHAOS"]:
                continue
            position = player.get("Position", "None")
            raw_name = player.get("Summoner Name", "Unknown")
            summoner_name = normalize_summoner_name(raw_name)
            if position in positions[team]:
                if position == "None":
                    positions[team][position].append(summoner_name)
                else:
                    positions[team][position] = summoner_name
        for team in ["ORDER", "CHAOS"]:
            for pos in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]:
                if positions[team][pos] is None and positions[team]["None"]:
                    positions[team][pos] = positions[team]["None"].pop(0)
        player_mapping = {}
        order_positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        for i, pos in enumerate(order_positions):
            player_id = i + 1
            player_name = positions["ORDER"][pos]
            if player_name:
                player_mapping[player_name] = {"ID": player_id, "Team": "BLUE", "Position": pos}
        chaos_positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        for i, pos in enumerate(chaos_positions):
            player_id = i + 6
            player_name = positions["CHAOS"][pos]
            if player_name:
                player_mapping[player_name] = {"ID": player_id, "Team": "RED", "Position": pos}
        remaining_order = positions["ORDER"]["None"]
        remaining_chaos = positions["CHAOS"]["None"]
        for team, remaining in [("ORDER", remaining_order), ("CHAOS", remaining_chaos)]:
            team_display = "BLUE" if team == "ORDER" else "RED"
            start_id = 1 if team == "ORDER" else 6
            positions_list = order_positions if team == "ORDER" else chaos_positions
            for player_name in remaining:
                for i, pos in enumerate(positions_list):
                    player_id = start_id + i
                    if not any(data.get("ID") == player_id for data in player_mapping.values()):
                        player_mapping[player_name] = {"ID": player_id, "Team": team_display, "Position": "Unknown"}
                        break
        self.player_mapping = player_mapping

    def process_events_with_player_ids(self, events_data):
        """이벤트 데이터의 킬러, 피해자, 어시스터 필드를 매핑 번호로 변환합니다."""
        processed_events = []
        for event in events_data:
            processed_event = event.copy()
            event_time_sec = float(event.get("EventTime", 0.0))
            processed_event["EventTime"] = math.floor(event_time_sec * 1000)
            
            killer_name = normalize_summoner_name(event.get("KillerName", "Unknown"))
            if killer_name in self.player_mapping:
                processed_event["KillerName"] = self.player_mapping[killer_name]["ID"]
            
            victim_name = normalize_summoner_name(event.get("VictimName", "Unknown"))
            if victim_name in self.player_mapping:
                processed_event["VictimName"] = self.player_mapping[victim_name]["ID"]
            
            assisters = event.get("Assisters", [])
            processed_assisters = []
            for assister in assisters:
                norm_assister = normalize_summoner_name(assister)
                if norm_assister in self.player_mapping:
                    processed_assisters.append(self.player_mapping[norm_assister]["ID"])
                else:
                    processed_assisters.append(assister)
            processed_event["Assisters"] = processed_assisters
            processed_events.append(processed_event)
        return processed_events

    def _overwrite_csv(self, df, filename):
        df.to_csv(self.game_folder / filename, index=False, encoding='utf-8-sig')

    def save_timing_data(self, request_time_ms, game_time_ms, time_difference_ms, start_time):
        """첫 번째 응답의 타이밍 정보를 CSV로 저장"""
        try:
            timing_data = [{
                "request_time_ms": request_time_ms,
                "game_duration_ms": game_time_ms,
                "game_start_time": time_difference_ms,
                "start_time": start_time
            }]
            df_timing = pd.DataFrame(timing_data)
            self._overwrite_csv(df_timing, "game_time.csv")
            logging.info(f"타이밍 데이터 저장: 시스템시간={request_time_ms}ms, 게임시간={game_time_ms}ms, 차이={time_difference_ms}ms")
        except Exception as e:
            logging.error(f"타이밍 데이터 저장 실패: {e}")

# 전역 변수로 이전 상태 저장
_previous_game_time: Optional[float] = None
_previous_global_time: Optional[int] = None  
_pause_start_time: Optional[int] = None
_is_paused: bool = False
_pause_records = []

def check_pause_and_record(cur_time: float, cur_global_time: int, game_folder: str) -> Dict[str, Any]:
    """
    게임 pause를 감지하고 기록하는 함수
    
    Args:
        cur_time: 현재 게임 시간 (gameTime)
        cur_global_time: 현재 글로벌 시간 (time.time_ns() // 1_000_000)
        game_folder: CSV 파일을 저장할 경로
        
    Returns:
        Dict: pause 상태 정보
    """
    global _previous_game_time, _previous_global_time, _pause_start_time, _is_paused, _pause_records
    
    result = {
        'is_paused': False,
        'pause_started': False,
        'pause_ended': False,
        'pause_duration_ms': 0,
        'total_pauses': len(_pause_records)
    }
    
    try:
        # 첫 번째 호출인 경우 초기화
        if _previous_game_time is None:
            _previous_game_time = cur_time
            _previous_global_time = cur_global_time
            logging.debug(f"Pause 감지 시작. 초기 게임 시간: {cur_time}")
            return result
        
        # Pause 감지 로직
        if cur_time == _previous_game_time:
            # 게임 시간이 동일 = Pause 상태
            if not _is_paused:
                # Pause 시작
                _is_paused = True
                _pause_start_time = _previous_global_time
                result['pause_started'] = True
                logging.info(f"Pause 시작 감지. 게임 시간: {cur_time}, 시작 시간: {_pause_start_time}")
            
            result['is_paused'] = True
            
        else:
            # 게임 시간이 다름 = 정상 진행 또는 Pause 종료
            if _is_paused:
                # Pause 종료
                pause_end_time = _previous_global_time
                pause_duration = pause_end_time - _pause_start_time
                
                # Pause 기록 생성
                pause_record = {
                    'pause_start_time': _pause_start_time,
                    'pause_end_time': pause_end_time,
                    'pause_duration_ms': pause_duration,
                    'game_time_when_paused': _previous_game_time,
                    'game_time_when_resumed': cur_time
                }
                
                # 기록 저장
                _pause_records.append(pause_record)
                
                # CSV 파일에 기록
                _save_pause_to_csv(pause_record, game_folder)
                
                result['pause_ended'] = True
                result['pause_duration_ms'] = pause_duration
                
                logging.info(f"Pause 종료 감지. 지속 시간: {pause_duration}ms")
                
                # Pause 상태 초기화
                _is_paused = False
                _pause_start_time = None
            
            result['is_paused'] = False
        
        # 다음 비교를 위해 현재 값들 저장
        _previous_game_time = cur_time
        _previous_global_time = cur_global_time
        result['total_pauses'] = len(_pause_records)
        
    except Exception as e:
        logging.error(f"Pause 감지 중 오류 발생: {e}")
        import traceback
        logging.error(traceback.format_exc())
        
    return result

def _save_pause_to_csv(pause_record: Dict[str, Any], game_folder: str):
    """Pause 기록을 CSV 파일에 저장하는 내부 함수"""
    try:
        csv_file_path = os.path.join(game_folder, "pause_duration.csv")
        os.makedirs(game_folder, exist_ok=True)
        
        new_data = pd.DataFrame([pause_record])
        
        if os.path.exists(csv_file_path):
            try:
                existing_data = pd.read_csv(csv_file_path)
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                combined_data = new_data
                logging.warning(f"기존 CSV 파일을 읽을 수 없어 새로 생성합니다: {csv_file_path}")
        else:
            combined_data = new_data
            logging.info(f"새로운 pause_duration.csv 파일 생성: {csv_file_path}")
        
        combined_data.to_csv(csv_file_path, index=False, encoding='utf-8')
        logging.debug(f"Pause 기록 저장 완료: {pause_record}")
        
    except Exception as e:
        logging.error(f"CSV 파일 저장 중 오류: {e}")

def reset_pause_detector():
    """pause 감지기 상태 초기화 함수"""
    global _previous_game_time, _previous_global_time, _pause_start_time, _is_paused, _pause_records
    
    _previous_game_time = None
    _previous_global_time = None
    _pause_start_time = None
    _is_paused = False
    _pause_records.clear()
    logging.info("Pause 감지기 초기화 완료")

def collect_data_during_game(game_data_fetcher, first_game_data, game_folder):
    """
    첫 번째 200 응답 후 2초 간격으로 데이터 수집
    """
    logging.info("게임 데이터 수집 시작 (2초 간격)")
    
    data_storage = DataStorage(game_folder)

    # 데이터 누적 변수
    all_players_accumulated = []
    events_accumulated = []

    # 첫 번째 데이터 처리
    game_time = first_game_data.get("gameData", {}).get("gameTime", 0.0)
    all_players_accumulated = [
        {
            "GameTime": game_time,
            "Summoner Name": player.get("summonerName", "Unknown"),
            "Champion Name": player.get("championName", "Unknown"),
            "Level": player.get("level", 0),
            "Kills": player.get("scores", {}).get("kills", 0),
            "Deaths": player.get("scores", {}).get("deaths", 0),
            "Assists": player.get("scores", {}).get("assists", 0),
            "Creep Score": player.get("scores", {}).get("creepScore", 0),
            "Ward Score": player.get("scores", {}).get("wardScore", 0.0),
            "Team": player.get("team", "Unknown"),
            "Position": player.get("position", "None") or "None",
        }
        for player in first_game_data.get("allPlayers", [])
    ]
    
    events_accumulated = [
        {
            "EventName": event.get("EventName", "Unknown"),
            "EventTime": event.get("EventTime", 0.0),
            "KillerName": event.get("KillerName", "Unknown"),
            "VictimName": event.get("VictimName", "Unknown"),
            "Assisters": event.get("Assisters", []),
            "EventType": event.get("EventType", "Unknown"),
        }
        for event in first_game_data.get("events", {}).get("Events", [])
    ]

    logging.info(f"첫 번째 데이터 처리 완료 - 게임시간: {game_time:.1f}s")

    try:
        # 2초 간격으로 계속 데이터 수집 (서버 부하 감소)
        while True:
            time.sleep(2.0)
            
            game_data = game_data_fetcher.get_all_game_data()
            
            if game_data is None:
                logging.info("게임 데이터 더 이상 없음. 수집 종료.")
                break

            # 현재 게임 시간
            game_time = game_data.get("gameData", {}).get("gameTime", 0.0)
            
            # pause 확인 
            cur_time = game_data.get("gameData", {}).get("gameTime", 0.0)
            cur_global_time = time.time_ns() // 1_000_000
            pause_result = check_pause_and_record(cur_time, cur_global_time, game_folder)

            if pause_result['pause_started']:
                logging.info("게임 Pause 시작됨")
            elif pause_result['pause_ended']:
                logging.info(f"게임 Pause 종료됨. 지속시간: {pause_result['pause_duration_ms']}ms")

            # 플레이어 데이터 갱신
            current_all_players = [
                {
                    "GameTime": game_time,
                    "Summoner Name": player.get("summonerName", "Unknown"),
                    "Champion Name": player.get("championName", "Unknown"),
                    "Level": player.get("level", 0),
                    "Kills": player.get("scores", {}).get("kills", 0),
                    "Deaths": player.get("scores", {}).get("deaths", 0),
                    "Assists": player.get("scores", {}).get("assists", 0),
                    "Creep Score": player.get("scores", {}).get("creepScore", 0),
                    "Ward Score": player.get("scores", {}).get("wardScore", 0.0),
                    "Team": player.get("team", "Unknown"),
                    "Position": player.get("position", "None") or "None",
                }
                for player in game_data.get("allPlayers", [])
            ]
            all_players_accumulated = current_all_players

            # 이벤트 데이터 갱신
            current_events = [
                {
                    "EventName": event.get("EventName", "Unknown"),
                    "EventTime": event.get("EventTime", 0.0),
                    "KillerName": event.get("KillerName", "Unknown"),
                    "VictimName": event.get("VictimName", "Unknown"),
                    "Assisters": event.get("Assisters", []),
                    "EventType": event.get("EventType", "Unknown"),
                }
                for event in game_data.get("events", {}).get("Events", [])
            ]
            events_accumulated = current_events

    except Exception as e:
        logging.error(f"데이터 수집 중 오류 발생: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        try:
            # 수집 종료 시간 기록
            end_time = time.time_ns() // 1_000_000
            timing_file = os.path.join(game_folder, "game_time.csv")
            
            if os.path.exists(timing_file):
                data = pd.read_csv(timing_file)
                data['collect_end_time'] = end_time
                data['duration'] = end_time - data['game_start_time']
                data.to_csv(timing_file, index=False)
            
            # 최종 데이터 저장
            data_storage.save_data_to_csv(all_players_accumulated, events_accumulated)
            logging.info(f"데이터 수집 완료. 저장 위치: {game_folder}")
            logging.info("저장된 파일: event.csv, info.csv, game_time.csv")
            
            # 데이터 압축 및 전송
            compress_and_send_files(str(game_folder))
            
        except Exception as e:
            logging.error(f"데이터 저장 중 오류: {e}")

def main():
    """
    메인 함수 - 전체 프로그램 흐름 제어
    1. 초기 설정 및 중복 실행 방지 (락 파일 사용)
    2. API를 통한 게임 감지
    3. 2초 간격 데이터 수집
    4. 게임 종료 후 다음 게임 대기
    """
    
    background_mode = "--background" in sys.argv
    
    logging.info(f"LOL Data Collector 시작. 백그라운드 모드: {background_mode}")
    
    # 팀 정보 설정
    get_team_info()
    logging.info(f"모니터링 설정 완료. 팀명: {TEAM_NAME}, 라벨: {TEAM_LABEL}")

    # API 기반 데이터 수집기 생성
    game_data_fetcher = GameDataFetcher()

    try:
        while True:
            try:
                logging.info("=" * 60)
                logging.info("새 게임 감지 시작")
                
                # pause 감지기 초기화 (새 게임 시작 시)
                reset_pause_detector()
                
                # 1단계: 첫 번째 성공적인 API 응답 대기 및 타이밍 정보 수집
                result = game_data_fetcher.wait_for_first_response()
                
                if result[0] is not None:  # game_data가 None이 아닌 경우
                    game_data, request_time_ms, game_time_ms, time_difference_ms, start_time = result
                    logging.info("첫 번째 응답 수신! 데이터 수집 시작")
                    
                    # 게임 폴더 생성
                    date_str = datetime.now().strftime("%Y%m%d_%H%M")
                    app_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Local", "LOLDataCollector")
                    os.makedirs(app_data_path, exist_ok=True)
                    game_folder = Path(os.path.join(app_data_path, f'match_{TEAM_NAME}_{TEAM_LABEL}_{date_str}'))
                    game_folder.mkdir(parents=True, exist_ok=True)
                    
                    # 타이밍 정보 저장
                    timing_storage = DataStorage(game_folder)
                    timing_storage.save_timing_data(request_time_ms, game_time_ms, time_difference_ms, start_time)
                    
                    # 2단계: API 기반 2초 간격 데이터 수집
                    collect_data_during_game(game_data_fetcher, game_data, game_folder)
                    
                    logging.info("게임 종료. 다음 게임 대기 중...")
                else:
                    logging.info("첫 번째 응답 감지 실패. 10초 후 재시도...")
                    time.sleep(10)
                    
            except Exception as e:
                logging.error(f"메인 루프 오류: {e}")
                import traceback
                logging.error(traceback.format_exc())
                logging.info("30초 후 재시도...")
                time.sleep(30)
                
    except KeyboardInterrupt:
        logging.info("사용자에 의해 프로그램 종료")
    finally:
        logging.info("프로그램 종료 완료")

if __name__ == "__main__":
    # 관리자 권한 요구 제거 - 일반 사용자 권한으로 실행
    # 이를 통해 대회 환경에서의 보안 정책 위반 방지
    logging.info("일반 사용자 권한으로 실행 중...")
    main()
