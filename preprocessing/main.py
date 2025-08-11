import os
import sys
from Research.project.Analysis.config import (
    DATASET_CSV, VOICE_PARAMS, SEQUENCE_PARAMS, 
    FEATURE_PARAMS, TEAM_COLORS, PATH_FORMAT
)
from utils import reset_gold_list
from voice_processor import VoiceProcessor
from data_processor import IngameDataProcessor
from event_classifier import EventClassifier
from sequence_generator import SequenceGenerator
from feature_extractor import Feature_Extractor_event
from dataset_builder import EventDatasetBuilder

"""
메인 실행 스크립트 - 클래스 인스턴스를 생성하고 유기적으로 연결
"""


def process_game(game_num):
    """
    단일 게임 데이터 처리
    
    Args:
        game_num: 처리할 게임 번호
    """
    print(f"\n{'='*20}")
    print(f"Processing game {game_num}...")
    print(f"{'='*20}")
    
    # 파일 경로 설정
    voice_root_directory = PATH_FORMAT["voice_root"].format(game_num=game_num)
    voice_info_path = PATH_FORMAT["voice_info"].format(game_num=game_num)
    ingame_json_path = PATH_FORMAT["ingame_json"].format(game_num=game_num)
    csv_filename = PATH_FORMAT["csv_filename"].format(game_num=game_num)
    
    # 절대 경로 출력
    print(f"Using voice directory: {os.path.abspath(voice_root_directory)}")
    print(f"Using voice info file: {os.path.abspath(voice_info_path)}")
    print(f"Using ingame JSON file: {os.path.abspath(ingame_json_path)}")
    print(f"Voice flags CSV file: {os.path.abspath(csv_filename)}")

    # 1. 음성 데이터 처리
    print("\n1. 음성 데이터 처리 시작")
    voice_processor = VoiceProcessor(
        voice_root_directory, 
        sr=VOICE_PARAMS["sr"], 
        hop_length=VOICE_PARAMS["hop_length"], 
        csv_base_filename=csv_filename
    )
    
    if os.path.exists(csv_filename):
        print(f"Loading existing voice flags from {os.path.abspath(csv_filename)}")
        voice_processor.load_voiced_flags(csv_filename)
    else:
        print(f"Processing audio files to extract voice flags")
        voice_processor.process_files_parallel()
        csv_filename = voice_processor.save_voiced_flags_csv()
        print(f"Saved voice flags to {os.path.abspath(csv_filename)}")
    
    print("Segmenting utterances")
    voice_results = voice_processor.process_utterances()
    
    # 2. 인게임 데이터 처리
    print("\n2. 인게임 데이터 처리 시작")
    ingame_processor = IngameDataProcessor(ingame_json_path, voice_info_path)
    ingame_processor.load_ingame_data()
    ingame_processor.load_voice_start_time()
    offset = ingame_processor.compute_offset()
    
    if offset is None:
        print(f"Error: Could not compute offset for game {game_num}. Skipping...")
        return False
    
    # 3. 이벤트 분류
    print("\n3. 이벤트 분류 시작")
    # 팀 색상 설정
    team_color = TEAM_COLORS.get(game_num, 1)  # 기본값은 Blue 팀 (1)
    print(f"Using team color: {'Blue' if team_color == 1 else 'Red'} (code: {team_color})")
    
    # gold_list 초기화 (이전 게임 데이터가 남아 있을 수 있음)
    reset_gold_list()
    
    # 인게임 이벤트 분류
    event_classifier = EventClassifier(ingame_processor.full_data, team_color)
    event_data = event_classifier.classify()     
    
    # 4. 이벤트 시퀀스 추출 
    print("\n4. 이벤트 시퀀스 생성 시작")
    sequence_generator = SequenceGenerator(voice_results, offset, team_color)
    event_sequences = sequence_generator.extract_window_sliding(
        event_data, 
        window_size=SEQUENCE_PARAMS["window_size"], 
        step_size=SEQUENCE_PARAMS["step_size"]
    )
    print(f"Generated {len(event_sequences)} event sequences")
    
    # 5. 패턴 추출 및 특성 추출
    print("\n5. 특성 추출 시작")
    event_feature_extractor = Feature_Extractor_event(event_sequences)
    event_features = event_feature_extractor.extract_features(FEATURE_PARAMS["CI"])
    print(f"Extracted features from {len(event_features)} event sequences")
    
    # 6. 데이터셋 구축
    print("\n6. 데이터셋 구축 시작")
    print(f"Dataset CSV file: {os.path.abspath(DATASET_CSV)}")
    event_builder = EventDatasetBuilder(DATASET_CSV, game_num)
    event_builder.add_features(event_features)
    
    print(f"Completed processing for game {game_num}")
    return True

def main():
    """메인 함수 - 모든 게임 처리 및 데이터셋 검사"""
    successful_games = 0
    
    # 현재 작업 디렉토리 출력
    print(f"Current working directory: {os.getcwd()}")
    print(f"Dataset will be saved to: {os.path.abspath(DATASET_CSV)}")
    
    # game nubmer loop 
    start_game = 1
    end_game = 4
    
    # 명령줄 인수 처리
    # python main.py 2 4
    # sys.argv[main.py, 2, 4]

    if len(sys.argv) > 1:
        try:
            if len(sys.argv) == 2:
                # 단일 게임만 처리
                start_game = end_game = int(sys.argv[1])
                print(f"Processing single game: {start_game}")
            elif len(sys.argv) >= 3:
                # 범위 지정
                start_game = int(sys.argv[1])
                end_game = int(sys.argv[2])
                print(f"Processing games from {start_game} to {end_game}")
        except ValueError:
            print("Warning: Invalid game number arguments. Using default range 1-4.")
            start_game, end_game = 1, 4
    else:
        print(f"Processing default game range: {start_game} to {end_game}")
    
    # argument없으면 default로 1-4 
    for game_num in range(start_game, end_game + 1):
        if process_game(game_num):
            successful_games += 1
    
    # 모든 게임 처리 후 최종 데이터셋 검사
    print("\n\nFinal dataset inspection:")
    event_builder = EventDatasetBuilder(DATASET_CSV, 0)  # game_num은 중요하지 않음
    event_builder.inspect_dataset()
    
    print(f"\nProcessing complete! Successfully processed {successful_games} out of {end_game - start_game + 1} games.")
    print(f"Final dataset saved at: {os.path.abspath(DATASET_CSV)}")

# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()