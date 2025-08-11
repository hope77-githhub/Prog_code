"""
이벤트 시퀀스 생성을 위한 SequenceGenerator 클래스
"""

import numpy as np

class SequenceGenerator:
    def __init__(self, voice_results, offset, team_color):
        """
        SequenceGenerator 클래스 초기화
        
        Args:
            voice_results: VoiceProcessor에서 생성한 utterance segmentation 결과 
                          (예: {'top1': [...], 'jug4': [...], ...})
            offset: 인게임 데이터와 음성 데이터 간 시간 차 보정값 (밀리초 단위)
            team_color: 팀 번호 (1: Blue, 2: Red)
        """
        self.voice_results = voice_results
        self.offset = offset
        self.team_color = team_color

    def extract_window_sliding(self, event_data, window_size=30000, step_size=1000):
        """
        이벤트 기준으로 슬라이싱된 데이터를 기반으로, 지정한 윈도우 크기와 슬라이딩 간격을 사용하여
        데이터 증강을 진행합니다.
        
        Args:
            event_data: 이벤트 기준 데이터 (기존과 동일 형식)
            window_size: 윈도우의 길이 (밀리초 단위, 기본값: 30000)
            step_size: 슬라이딩 간격 (밀리초 단위, 기본값: 1000)
            
        Returns:
            list: 증강된 이벤트 시퀀스 리스트
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