"""
데이터셋 구축을 위한 EventDatasetBuilder 클래스
"""

import os
import pandas as pd
from itertools import product
from Research.project.Analysis.config import SPEAKERS, EVENT_LABEL_MAP

class EventDatasetBuilder:
    def __init__(self, csv_file: str, game_num: int):
        """
        EventDatasetBuilder 클래스 초기화
        
        Args:
            csv_file: 이벤트 데이터셋을 저장할 CSV 파일 경로
            game_num: 이 클래스 인스턴스가 담당할 게임(매치) 번호
        """
        self.csv_file = csv_file
        self.game_num = game_num
        
        # 파일명 출력
        print(f"Dataset will be saved to: {os.path.abspath(csv_file)}")
        
        # 기준 발화자 순서
        self.speakers = SPEAKERS
        
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
        self.event_label_map = EVENT_LABEL_MAP
        
        # csv_file이 이미 존재하면 읽어서 DataFrame으로 로드, 아니면 빈 DataFrame 생성
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            print(f"Loaded existing dataset from: {os.path.abspath(csv_file)}")
        else:
            self.df = pd.DataFrame(columns=self.all_columns)
            print(f"Creating new dataset file: {os.path.abspath(csv_file)}")
    
    def add_features(self, event_features: list):
        """
        이벤트별 feature 딕셔너리 리스트를 받아 CSV 파일에 추가 저장합니다.
        
        Args:
            event_features (list): Feature_Extractor_event.extract_features()의 결과 리스트
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
                if p3[0] != p3[1] and p3[1] != p3[2]:  # 연속된 동일 화자가 없는 경우
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
            print(f"Added {len(rows)} new rows to dataset: {os.path.abspath(self.csv_file)}")
        else:
            print("No valid data to add to dataset")
    
    def inspect_dataset(self):
        """
        CSV 파일에 저장된 데이터셋 정보를 확인합니다.
        """
        if os.path.exists(self.csv_file):
            df = pd.read_csv(self.csv_file)
            print(f"Inspecting dataset: {os.path.abspath(self.csv_file)}")
            print("Dataset summary:")
            print(f"- Total rows: {len(df)}")
            print(f"- Event distribution: {df['event_label'].value_counts().to_dict()}")
            print(f"- Games included: {sorted(df['game_num'].unique())}")
            print(f"- Total columns: {len(df.columns)}")
        else:
            print(f"CSV file '{os.path.abspath(self.csv_file)}' does not exist.")