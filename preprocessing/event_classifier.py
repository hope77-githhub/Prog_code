"""
인게임 이벤트 분류를 위한 EventClassifier 클래스
"""

from utils import gold_calculator
from Research.project.Analysis.config import ASSIST_GOLD_MAPPING

class EventClassifier:
    def __init__(self, loaded_data, team_color):
        """
        EventClassifier 초기화
        
        Args:
            loaded_data: 인게임 데이터의 JSON 객체 (특히 'frames' 포함)
            team_color: 팀 번호 (1: Blue, 2: Red)
        """
        self.loaded_data = loaded_data
        self.team_color = team_color
        self.event_data = {
            'CHAMPION_KILL': [],
            'ELITE_MONSTER_KILL': [],
            'BUILDING_KILL': [],
            'TURRET_PLATE_DESTROYED': []
        }
        self.assist_gold_mapping = ASSIST_GOLD_MAPPING

    def handle_champion_kill(self, event):
        """
        챔피언 처치 이벤트 처리
        
        Args:
            event: 이벤트 데이터 딕셔너리
        """
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
        """
        정예 몬스터 처치 이벤트 처리
        
        Args:
            event: 이벤트 데이터 딕셔너리
        """
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
        """
        건물 파괴 이벤트 처리
        
        Args:
            event: 이벤트 데이터 딕셔너리
        """
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
        """
        포탑 방어막 파괴 이벤트 처리
        
        Args:
            event: 이벤트 데이터 딕셔너리
        """
        killer_id = event.get('killerId', 0)
        success = 1 if ((self.team_color == 1 and killer_id < 6) or (self.team_color == 2 and killer_id >= 6)) else 0
        gold = 125
        self.event_data['TURRET_PLATE_DESTROYED'].append({
            'time': event['timestamp'],
            'gold': gold,
            'success': success
        })
        
    def classify(self):
        """
        모든 이벤트를 분류하고 처리
        
        Returns:
            dict: 분류된 이벤트 데이터
        """
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