"""
음성 데이터 처리를 위한 VoiceProcessor 클래스
"""

import os
import json
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class VoiceProcessor:
    def __init__(self, voice_root_directory, sr=22050, hop_length=512, csv_base_filename="voiced_flags"):
        """
        음성 데이터 처리 클래스 초기화
        
        Args:
            voice_root_directory: 음성 파일이 있는 루트 디렉토리
            sr: 샘플링 레이트 (기본값: 22050)
            hop_length: 호프 길이 (기본값: 512)
            csv_base_filename: 음성 플래그를 저장할 CSV 파일 이름 (기본값: "voiced_flags")
        """
        self.voice_root_directory = voice_root_directory
        self.sr = sr
        self.hop_length = hop_length
        self.csv_base_filename = csv_base_filename
        self.voiced_flags = {}  # {파일명: voiced_flag (numpy array)}
        self.results = {}       # {파일명: utterance segmentation 결과 (리스트)}

    def get_audio_file_paths(self, extension=".wav"):
        """
        지정된 디렉토리에서 특정 확장자를 가진 모든 오디오 파일 경로 반환
        
        Args:
            extension: 찾을 파일 확장자 (기본값: ".wav")
            
        Returns:
            list: 오디오 파일 경로 리스트
        """
        file_paths = []
        for root, _, files in os.walk(self.voice_root_directory):
            for file in files:
                if file.endswith(extension):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def voiced_check(self, file_name, file_path):
        """
        오디오 파일에서 음성/비음성 플래그 추출
        
        Args:
            file_name: 오디오 파일 이름
            file_path: 오디오 파일 경로
            
        Returns:
            numpy.ndarray: 음성/비음성 플래그 배열
        """
        y, sr = librosa.load(file_path, sr=self.sr)
        print(f'{file_name} Sampling Rate is.. {sr}')
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, sr=sr,
            fmin=librosa.note_to_hz('C2'), #E2
            fmax=librosa.note_to_hz('C5') #C5
        )
        return voiced_flag

    def process_file(self, path):
        """
        단일 오디오 파일 처리
        
        Args:
            path: 오디오 파일 경로
            
        Returns:
            tuple: (파일이름, voiced_flag 배열)
        """
        file_name = os.path.splitext(os.path.basename(path))[0]
        voiced_flag = self.voiced_check(file_name, path)
        return file_name, voiced_flag

    def process_files_parallel(self, num_workers=None):
        """
        병렬 처리를 통해 모든 오디오 파일을 처리
        
        Args:
            num_workers: 병렬 처리에 사용할 워커 수 (None이면 자동 설정)
            
        Returns:
            dict: {파일명: voiced_flag 배열} 형태의 결과
        """
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
        """
        음성 플래그를 CSV 파일로 저장
        
        Args:
            starting_number: CSV 파일명 번호 시작점
            
        Returns:
            str: 저장된 CSV 파일명
        """
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
        """
        저장된 CSV 파일에서 음성 플래그 로드
        
        Args:
            csv_filename: 로드할 CSV 파일명
            
        Returns:
            dict: 로드된 음성 플래그 {파일명: voiced_flag 배열}
        """
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

    # threshold is heuristic hyperparameter 
    
    def utterance_segmentation(self, voice_data, threshold=100): 
        """
        음성 데이터에서 발화 구간 세그먼트 추출
        
        Args:
            voice_data: 음성/비음성 플래그 배열
            threshold: 무음 구간 임계값 (기본값: 100 프레임)
            
        Returns:
            list: 발화 구간 리스트 [시작 프레임, 종료 프레임, 구간 길이]
        """
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
        """
        프레임 인덱스를 시간(밀리초)으로 변환
        
        Args:
            sorted_range: 프레임 인덱스 범위 리스트 [[시작, 종료, 길이], ...]
            
        Returns:
            list: 시간 범위 리스트 [{"start": 시작시간(ms), "end": 종료시간(ms)}, ...]
        """
        sorted_time = []
        for r in sorted_range:
            # ms 단위로 변환하여 ms단위로 모든 데이터를 통일. (1000ms = 1s)
            start_time = round(librosa.frames_to_time(r[0], sr=self.sr, hop_length=self.hop_length) * 1000)
            end_time = round(librosa.frames_to_time(r[1], sr=self.sr, hop_length=self.hop_length) * 1000) 
            sorted_time.append({"start": start_time, "end": end_time})
        return sorted_time

    def process_utterances(self, threshold=100):
        """
        모든 음성 파일에 대해 발화 구간 추출
        
        Args:
            threshold: 무음 구간 임계값 (기본값: 100 프레임)
            
        Returns:
            dict: {파일명: 발화 구간 시간 리스트}
        """
        results = {}
        for file_name, flag in self.voiced_flags.items():
            sorted_utterance = self.utterance_segmentation(flag, threshold=threshold)
            sorted_time = self.frame_to_time(sorted_utterance)
            results[file_name] = sorted_time
        self.results = results
        return results