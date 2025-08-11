import discord
import asyncio
import os
import time
import logging
from datetime import datetime
from discord.sinks.core import Sink, Filters, default_filters  # overriding
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
import sys
import platform
import uuid
from collections import defaultdict

# Windows 전용 이벤트 루프 정책 설정
if sys.platform == "win32" or platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 로깅 설정 (디버깅용)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from datetime import datetime

# --- 설정 파일 경로 및 전역 변수 ---
CONFIG_FILE = os.path.join(os.path.expanduser("~"), "lol_data_collector_config.json")
TEAM_NAME = ""
TEAM_LABEL = ""
LABEL_DESCRIPTIONS = {
    1: "1군 LCK",
    2: "2군 CL",
    3: "3군 아카데미",
    4: "아카데미",
}


def load_config():
    """설정 파일에서 팀명과 라벨 로드"""
    global TEAM_NAME, TEAM_LABEL
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
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
        config = {"team_name": TEAM_NAME, "team_label": TEAM_LABEL}
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)
        logging.info(f"설정 파일 저장 완료: {CONFIG_FILE}")
        return True
    except Exception as e:
        logging.error(f"설정 파일 저장 실패: {e}")
        return False


def get_team_info():
    """최초 실행 시 팀명과 라벨을 입력받습니다."""
    global TEAM_NAME, TEAM_LABEL
    if load_config():
        return

    print("크레이그봇 첫 발화 로그 수집 프로그램을 시작합니다.")
    TEAM_NAME = input("데이터 수집자(팀)명을 입력해주세요: ")
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

    print(
        f"\n수집자: {TEAM_NAME}, 라벨: {TEAM_LABEL} ({LABEL_DESCRIPTIONS[int(TEAM_LABEL)]})"
    )
    save_config()


get_team_info()


def send_log_file(file_path: str):
    """저장된 txt 로그 파일을 이메일로 전송 (동기, 호출부에서 to_thread로 비동기화)"""
    try:
        sender_email = "datacollection.ixlab@gmail.com"
        sender_password = ""
        receiver_email = "datacollection.ixlab@gmail.com"

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Date"] = formatdate(localtime=True)
        msg["Subject"] = f"Log File - {os.path.basename(file_path)}"
        msg.attach(
            MIMEText(
                f"Attached is the log file: {os.path.basename(file_path)}", "plain"
            )
        )

        with open(file_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition", "attachment", filename=os.path.basename(file_path)
            )
            msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logging.info(f"Successfully sent log file to {receiver_email}")
    except Exception as e:
        logging.error(f"Failed to send log file: {e}")


# 필요한 설정 가져오기
class Config:
    token = ""  # 환경변수에서 토큰 가져오기 권장
    guild = None


# 봇 인스턴스 생성 (모든 권한 부여)
intents = discord.Intents.all()
bot = discord.Bot(intents=intents)

# 길드별 동시성 제어 락
guild_locks = defaultdict(asyncio.Lock)

# 연결 정보를 저장할 딕셔너리
# connections[guild_id] = {
#   "voice": VoiceClient | None,
#   "detecting": bool,
#   "channel": VoiceChannel,
#   "sink": VoiceDetectionSink | None,
#   "auto_joined": bool,
#   "waiting_for_craig": bool,
#   "session_id": str
# }
connections = {}

# 첫 발화 감지 세션 정보 저장용 (채널 단위)
# first_speech_sessions[channel_id] = {...}
first_speech_sessions = {}

# Craig bot 설정 (Craig bot의 ID들)
CRAIG_BOT_IDS = [
    272937604339466240,  # Craig#7720 (주요 Craig bot)
    # 다른 Craig bot ID가 있다면 여기에 추가
]


# 저장 경로 설정
def get_app_data_path():
    """LOLDataCollector와 동일한 경로 설정"""
    app_data_path = os.path.join(
        os.path.expanduser("~"), "AppData", "Local", "LOLDataCollector"
    )
    os.makedirs(app_data_path, exist_ok=True)
    return app_data_path


# 첫 발화 감지 로그를 파일로 저장하는 함수
def save_first_speech_log(
    guild_id, channel_id, first_speaker_id, detection_time_ns, channel_name
):
    """첫 발화 감지 정보를 txt 파일로 저장"""
    try:
        app_data_path = get_app_data_path()
        os.makedirs(app_data_path, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        timestamp_ms = detection_time_ns // 1_000_000

        # 파일명 충돌 방지를 위해 guild/channel 포함
        filename = (
            f"match_{TEAM_NAME}_{TEAM_LABEL}_{guild_id}_{channel_id}_{date_str}_{timestamp_ms}.txt"
        )
        filepath = os.path.join(app_data_path, filename)

        detection_time_seconds = detection_time_ns / 1_000_000_000
        utc_time = datetime.utcfromtimestamp(detection_time_seconds).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"서버 ID: {guild_id}\n")
            f.write(f"채널 ID: {channel_id}\n")
            f.write(f"첫 발화감지 타임스탬프(ns): {detection_time_ns}\n")
            f.write(f"첫 발화자 ID: {first_speaker_id}\n")
            f.write(f"UTC 시간: {utc_time}\n")

        logging.info(f"첫 발화 감지 로그 저장 완료: {filepath}")
        print(f"첫 발화 로그 저장: {filename} (채널: {channel_name})")

        # 메일 전송은 스레드로 비동기 처리
        asyncio.create_task(asyncio.to_thread(send_log_file, filepath))
    except Exception as e:
        logging.error(f"첫 발화 로그 저장 실패: {e}")


# Craig bot 감지 함수
def is_craig_bot(member):
    """Craig bot인지 확인하는 함수"""
    if not member.bot:
        return False

    if member.id in CRAIG_BOT_IDS:
        logger.info(f"Craig bot 감지됨 (ID 매칭): {member.name} (ID: {member.id})")
        return True

    name_lower = member.name.lower()
    display_name_lower = (
        member.display_name.lower() if hasattr(member, "display_name") else ""
    )

    craig_indicators = [
        "craig",
        "[recording] craig",
        "![recording] craig",
        "recording craig",
        "craig recording",
        "recording",
    ]

    for indicator in craig_indicators:
        if indicator in name_lower or indicator in display_name_lower:
            logger.info(
                f"Craig bot 감지됨 - 이름: '{member.name}', 표시명: '{getattr(member, 'display_name', 'N/A')}', 매칭: '{indicator}'"
            )
            return True

    return False


# 커스텀 음성 감지 Sink 클래스
class VoiceDetectionSink(Sink):
    """첫 발화자만 감지하는 초경량 Sink"""

    def __init__(self, bot, *, filters=None):
        if filters is None:
            filters = default_filters.copy()

        try:
            super().__init__(filters=filters)
            self.bot = bot

            # 첫 발화자 정보만 저장 (단순화)
            self.first_speaker_id = None
            self.first_speaker_time_ns = None
            self.first_speech_processed = False

            # 디버그/정리용 기본 필드 보강
            self.user_cache = {}
            self.last_detection = {}
            self.detection_cooldown = 0.0
            self.cleanup_interval = 300.0
            self.last_cleanup = time.time()
            self.first_speech_detected = False

            self.vc = None  # VoiceClient 참조

            logger.info(f"VoiceDetectionSink 초기화 완료 - 첫 발화자 전용 모드")

        except Exception as e:
            logger.error(f"VoiceDetectionSink 초기화 실패: {e}")
            raise

    def init(self, vc):
        """Sink 초기화 - VoiceClient 연결 시 호출됨"""
        try:
            super().init(vc)
            self.vc = vc
            logger.info(f"VoiceClient 연결 완료. 채널: {vc.channel.name}")
        except Exception as e:
            logger.error(f"VoiceClient 초기화 실패: {e}")
            raise

    def format_audio(self, file):
        """Discord.py Sink의 필수 메서드 (빈 구현)"""
        pass

    @Filters.container
    def write(self, data, user):
        """핵심 메서드: 첫 발화자만 감지하고 나머지는 모두 무시"""
        if self.first_speaker_id is not None:
            return

        detection_time_ns = time.time_ns()

        try:
            super().write(data, user)

            # 첫 발화자 확정 (한 번만 실행됨)
            self.first_speaker_id = user
            self.first_speaker_time_ns = detection_time_ns

            try:
                asyncio.run_coroutine_threadsafe(
                    self.process_first_speaker(user, detection_time_ns), self.bot.loop
                )
                logger.debug(f"첫 발화자 처리 태스크 전달 완료: {user}")
            except Exception as e:
                logger.error(f"첫 발화자 처리 태스크 생성 실패: {e}")

        except Exception as e:
            logger.error(f"write 메서드 오류: {e}")

    async def process_first_speaker(self, user_id, detection_time_ns):
        """첫 발화자 처리 - Craig bot 필터링 및 로그 저장"""
        try:
            # Craig bot 필터링
            if user_id in CRAIG_BOT_IDS:
                self.first_speaker_id = None
                self.first_speaker_time_ns = None
                return

            if self.first_speech_processed:
                return

            user = self.bot.get_user(user_id)
            user_name = user.display_name if user else f"Unknown User ({user_id})"

            channel_name = "Unknown"
            guild_id = None
            channel_id = None

            if self.vc and self.vc.channel:
                channel_id = self.vc.channel.id
                channel_name = self.vc.channel.name
                guild_id = self.vc.channel.guild.id

                if channel_id in first_speech_sessions:
                    session_info = first_speech_sessions[channel_id]

                    if not session_info.get("first_speech_detected", False):
                        self.first_speech_processed = True
                        self.first_speech_detected = True
                        session_info["first_speech_detected"] = True
                        session_info["first_speaker_id"] = user_id
                        session_info["first_speech_time"] = detection_time_ns

                        logger.info(
                            f"★★★ 첫 발화 감지 완료! 사용자: {user_name}, 채널: {channel_name}"
                        )
                        logger.info(f"첫 발화 시간(ns): {detection_time_ns}")

                        save_first_speech_log(
                            guild_id, channel_id, user_id, detection_time_ns, channel_name
                        )

                        # 현재 길드 세션 ID를 잡아와서 동일 세션에서만 auto_leave 수행
                        session_id = None
                        ci = connections.get(guild_id)
                        if ci:
                            session_id = ci.get("session_id")

                        logger.info(f"첫 발화 감지 완료로 자동 퇴장합니다: {channel_name}")
                        asyncio.create_task(
                            self.auto_leave_after_first_speech(
                                guild_id, channel_id, channel_name, session_id
                            )
                        )
                        return

            logger.info(f"첫 발화자 감지됨: {user_name} in {channel_name}")

        except Exception as e:
            logger.error(f"첫 발화자 처리 실패: {e}")

    async def auto_leave_after_first_speech(
        self, guild_id, channel_id, channel_name, session_id
    ):
        """첫 발화 감지 후 자동 퇴장 처리 (세션/채널 가드 포함)"""
        try:
            await asyncio.sleep(2)

            async with guild_locks[guild_id]:
                ci = connections.get(guild_id)
                if not ci:
                    logger.info(
                        f"[VOICE] auto_leave 스킵: 연결 정보 없음 ({channel_name})"
                    )
                    return

                # 동일 세션인지 확인 (세션 꼬임 방지)
                if session_id and ci.get("session_id") != session_id:
                    logger.info(
                        f"[VOICE] auto_leave 스킵: 세션 변경됨 ({channel_name})"
                    )
                    return

                voice = ci.get("voice")
                if not voice or not voice.channel or voice.channel.id != channel_id:
                    logger.info(
                        f"[VOICE] auto_leave 스킵: 이미 다른 채널/세션으로 전환됨 ({channel_name})"
                    )
                    return

                if ci.get("detecting", False):
                    logger.info("첫 발화 감지 완료로 인한 음성 감지 중지...")
                    try:
                        voice.stop_recording()
                    except Exception as e:
                        logger.warning(f"음성 감지 중지 실패 (이미 중지됨): {e}")

                logger.info(f"첫 발화 감지 완료로 인한 음성 채널 연결 해제: {channel_name}")
                try:
                    await voice.disconnect(force=True)
                except Exception as e:
                    logger.warning(f"음성 채널 연결 해제 실패 (이미 해제됨): {e}")

                # 연결 정보를 대기 상태로 변경
                ci["detecting"] = False
                ci["voice"] = None
                ci["sink"] = None
                ci["waiting_for_craig"] = True

                # 첫 발화 세션 정보 제거
                if channel_id in first_speech_sessions:
                    del first_speech_sessions[channel_id]
                    logger.info(f"첫 발화 세션 정보 제거 완료: {channel_name}")

                logger.info(
                    f"첫 발화 감지 후 자동 퇴장 처리 완료 (크레이그봇 대기 모드): {channel_name}"
                )

        except Exception as e:
            logger.error(f"첫 발화 후 자동 퇴장 처리 중 오류: {e}", exc_info=True)


async def follow_craig_bot(channel):
    """Craig bot이 입장한 채널에 자동으로 입장하여 음성 감지 시작"""
    guild = channel.guild
    guild_id = guild.id
    async with guild_locks[guild_id]:
        try:
            me = guild.me

            # ── 0) 권한/가시성/사용자 제한 검사 ───────────────────────────────
            perms = channel.permissions_for(me)
            if not perms.view_channel:
                logger.warning(f"[VOICE] 채널 가시성 부족(view_channel 없음): {channel.name}")
                return
            if not perms.connect:
                logger.warning(f"[VOICE] 채널 연결 권한(connect) 없음: {channel.name}")
                return
            if not perms.speak:
                logger.warning(
                    f"[VOICE] 말하기 권한(speak) 없음: {channel.name} (녹음만 해도 정책상 차단될 수 있음)"
                )

            # 사용자 수 제한 채널에서 슬랏이 꽉 찼다면 접속 불가
            if channel.user_limit and len(channel.members) >= channel.user_limit:
                logger.warning(f"[VOICE] 채널 정원 초과로 접속 불가: {channel.name}")
                return

            # ── 1) 기존 연결(또는 꼬인 상태) 정리 및 중복 연결 방지 ───────────
            if guild.voice_client:
                vc = guild.voice_client
                if vc.is_connected():
                    if vc.channel == channel:
                        logger.info(f"[VOICE] 이미 {channel.name}에 연결됨 → 재사용")
                    else:
                        logger.info(
                            f"[VOICE] 다른 채널에 연결되어 있어 먼저 해제: {vc.channel.name}"
                        )
                        try:
                            await vc.disconnect(force=True)
                        except Exception as e:
                            logger.warning(f"[VOICE] 기존 연결 강제 해제 실패: {e}")
                else:
                    try:
                        await vc.disconnect(force=True)
                    except:
                        pass

            # connections 보호: 활성 연결 중복 체크 (대기 모드는 재연결 허용)
            if guild_id in connections:
                ci = connections[guild_id]
                if ci.get("detecting", False) and ci.get("voice"):
                    logger.info(
                        f"[VOICE] 이미 활성 상태 연결 존재 → 입장 생략: {channel.name}"
                    )
                    return

            # ── 2) 안전한 연결 시도(타임아웃/재시도/backoff 포함) ─────────────
            async def connect_with_retry(ch, *, timeout=10, retries=2, backoff=2):
                last_exc = None
                for attempt in range(retries + 1):
                    try:
                        logger.info(
                            f"[VOICE] 채널 연결 시도 {attempt+1}/{retries+1}: {ch.name}"
                        )
                        vc = await ch.connect(timeout=timeout, reconnect=False)
                        return vc
                    except asyncio.TimeoutError as e:
                        last_exc = e
                        logger.warning(
                            f"[VOICE] 연결 타임아웃(≈{timeout}s). 재시도 대기 {backoff*(attempt+1)}s"
                        )
                        await asyncio.sleep(backoff * (attempt + 1))
                    except discord.ClientException as e:
                        if "Already connected" in str(e):
                            if ch.guild.voice_client and ch.guild.voice_client.is_connected():
                                return ch.guild.voice_client
                        last_exc = e
                        logger.warning(f"[VOICE] ClientException: {e}")
                        await asyncio.sleep(1.0)
                    except Exception as e:
                        last_exc = e
                        logger.error(f"[VOICE] 알 수 없는 연결 오류: {e}", exc_info=True)
                        await asyncio.sleep(1.5)
                raise last_exc

            try:
                voice = await connect_with_retry(channel)
            except Exception as e:
                logger.error(
                    f"[VOICE] 최종 연결 실패. 대기 모드로 전환: {channel.name} / 원인: {type(e).__name__}: {e}"
                )
                connections[guild_id] = {
                    "voice": None,
                    "detecting": False,
                    "channel": channel,
                    "sink": None,
                    "auto_joined": True,
                    "waiting_for_craig": True,
                    "session_id": uuid.uuid4().hex,  # 실패 세션도 식별자 부여
                }
                return

            # ── 3) Sink 준비 및 레코딩 시작 ───────────────────────────────────
            voice_sink = VoiceDetectionSink(bot)

            async def finished_callback(sink):
                logger.info("[VOICE] Craig bot 따라 입장한 음성 감지 세션 종료")

            try:
                voice.start_recording(voice_sink, finished_callback)
            except Exception as e:
                logger.error(f"[VOICE] start_recording 실패: {e}", exc_info=True)
                try:
                    await voice.disconnect(force=True)
                except:
                    pass
                connections[guild_id] = {
                    "voice": None,
                    "detecting": False,
                    "channel": channel,
                    "sink": None,
                    "auto_joined": True,
                    "waiting_for_craig": True,
                    "session_id": uuid.uuid4().hex,
                }
                return

            # ── 4) 연결/세션 상태 저장 ────────────────────────────────────────
            session_id = uuid.uuid4().hex
            connections[guild_id] = {
                "voice": voice,
                "detecting": True,
                "channel": channel,
                "sink": voice_sink,
                "auto_joined": True,
                "waiting_for_craig": False,
                "session_id": session_id,
            }

            first_speech_sessions[channel.id] = {
                "craig_join_time": time.time_ns(),
                "first_speech_detected": False,
                "channel_name": channel.name,
                "guild_id": guild_id,
            }

            logger.info(f"[VOICE] Craig bot 따라 음성 감지 시작 완료: {channel.name}")
            print(f"크레이그봇 추적: {channel.name} 채널에서 첫 발화 감지 시작")

        except Exception as e:
            logger.error(f"[VOICE] follow_craig_bot 처리 중 오류: {e}", exc_info=True)


async def follow_craig_bot_leave(channel):
    """Craig bot이 퇴장한 채널에서 자동으로 퇴장"""
    guild_id = channel.guild.id
    async with guild_locks[guild_id]:
        try:
            if guild_id not in connections:
                logger.info(f"Craig bot 퇴장 감지되었으나 연결 정보가 없음: {channel.name}")
                return

            connection_info = connections[guild_id]
            voice = connection_info.get("voice")

            if connection_info.get("waiting_for_craig", False):
                logger.info(f"크레이그봇 대기 상태에서 퇴장 이벤트 무시: {channel.name}")
                return

            if not voice or voice.channel != channel:
                logger.info(
                    f"Craig bot 퇴장 감지되었으나 음성 클라이언트 정보가 일치하지 않음: {channel.name}"
                )
                return

            if not connection_info.get("auto_joined", False):
                logger.info(
                    f"수동 입장한 채널이므로 Craig bot 퇴장 시 자동 퇴장하지 않습니다: {channel.name}"
                )
                return

            channel_name = voice.channel.name
            channel_id = voice.channel.id

            logger.info(f"Craig bot 퇴장으로 인한 정리 작업 시작: {channel_name}")

            if channel_id in first_speech_sessions:
                session_info = first_speech_sessions[channel_id]
                first_speech_detected = session_info.get("first_speech_detected", False)
                if first_speech_detected:
                    logger.info(f"첫 발화가 이미 감지된 세션의 Craig bot 퇴장: {channel_name}")
                else:
                    logger.info(f"첫 발화 감지 전 Craig bot 퇴장: {channel_name}")
                del first_speech_sessions[channel_id]
                logger.info(f"첫 발화 세션 정보 제거: {channel_name}")

            if connection_info.get("detecting", False):
                logger.info("Craig bot 퇴장으로 인한 음성 감지 중지 중...")
                try:
                    voice.stop_recording()
                except Exception as e:
                    logger.warning(f"음성 감지 중지 실패 (이미 중지됨): {e}")

            logger.info(f"Craig bot 퇴장으로 인한 음성 채널 연결 해제 중: {channel_name}")
            try:
                await voice.disconnect(force=True)
            except Exception as e:
                logger.warning(f"음성 채널 연결 해제 실패 (이미 해제됨): {e}")

            connection_info["detecting"] = False
            connection_info["voice"] = None
            connection_info["sink"] = None
            connection_info["waiting_for_craig"] = True

            logger.info(f"Craig bot 퇴장으로 인한 정리 작업 완료 (대기 모드): {channel_name}")
            print(f"크레이그봇 퇴장: {channel_name} 채널에서 대기 모드로 전환")

        except Exception as e:
            logger.error(f"Craig bot 따라 퇴장 중 오류: {e}", exc_info=True)


# 서버 참여 이벤트 처리
@bot.event
async def on_guild_join(guild):
    print(f"새로운 서버에 참여했습니다: {guild.name} (ID: {guild.id})")
    Config.guild = guild.id


# 봇 시작 이벤트 처리
@bot.event
async def on_ready():
    logger.info(f"{bot.user}로 로그인했습니다!")
    print(f"{bot.user}로 로그인했습니다!")
    print(f"봇 ID: {bot.user.id}")

    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching, name="Craig Bot 지속 추적 & 첫 발화 감지"
        ),
        status="online",
    )

    app_data_path = get_app_data_path()
    print(f"저장 경로: {app_data_path}")

    for guild in bot.guilds:
        print(f'서버 "{guild.name}" (ID: {guild.id})에 연결되었습니다')
        if Config.guild is None:
            Config.guild = guild.id


# 음성 상태 변경 이벤트 감지
@bot.event
async def on_voice_state_update(member, before, after):
    if is_craig_bot(member):
        if before.channel is None and after.channel is not None:
            print(f"Craig bot ({member.name}) 입장 감지: {after.channel.name}")
            logger.info(f"Craig bot 입장 감지: {member.name} → {after.channel.name}")
            await follow_craig_bot(after.channel)

        elif before.channel is not None and after.channel is None:
            print(f"Craig bot ({member.name}) 퇴장 감지: {before.channel.name}")
            logger.info(f"Craig bot 퇴장 감지: {member.name} ← {before.channel.name}")
            await follow_craig_bot_leave(before.channel)

        elif (
            before.channel != after.channel
            and before.channel is not None
            and after.channel is not None
        ):
            print(
                f"Craig bot ({member.name}) 채널 이동: {before.channel.name} → {after.channel.name}"
            )
            logger.info(
                f"Craig bot 채널 이동: {member.name} {before.channel.name} → {after.channel.name}"
            )
            await follow_craig_bot_leave(before.channel)
            await asyncio.sleep(0.6)  # 게이트웨이 정리 시간 살짝 부여
            await follow_craig_bot(after.channel)

    else:
        if before.channel is None and after.channel is not None:
            print(f"{member.name}님이 {after.channel.name} 채널에 입장했습니다.")

        elif before.channel is not None and after.channel is None:
            print(f"{member.name}님이 {before.channel.name} 채널에서 퇴장했습니다.")

            for guild_id, connection_info in list(connections.items()):
                voice = connection_info.get("voice")
                if (
                    voice
                    and voice.channel == before.channel
                    and len(voice.channel.members) <= 1
                    and not connection_info.get("auto_joined", False)
                    and not connection_info.get("waiting_for_craig", False)
                ):
                    if connection_info.get("detecting", False):
                        voice.stop_recording()
                    await voice.disconnect(force=True)
                    del connections[guild_id]
                    print(f"채널 {before.channel.name}에 사람이 없어서 봇이 퇴장했습니다.")


@bot.event
async def on_error(event, *args, **kwargs):
    logger.error(f"봇 오류 발생: {event}", exc_info=True)


@bot.slash_command(description="현재 Craig bot 첫 발화 세션 정보를 확인합니다")
async def speech_status(ctx):
    try:
        status_parts = []

        if first_speech_sessions:
            status_parts.append("**활성 첫 발화 세션**")
            for channel_id, session_info in first_speech_sessions.items():
                channel_name = session_info.get("channel_name", "Unknown")
                craig_join_time = session_info.get("craig_join_time", 0)
                first_speech_detected = session_info.get("first_speech_detected", False)

                elapsed_seconds = (time.time_ns() - craig_join_time) / 1_000_000_000
                elapsed_minutes = elapsed_seconds / 60

                status = "첫 발화 감지됨" if first_speech_detected else "첫 발화 대기 중"
                status_parts.append(f"• {channel_name}: {status} ({elapsed_minutes:.1f}분 경과)")
        else:
            status_parts.append("**활성 첫 발화 세션**: 없음")

        waiting_connections = []
        active_connections = []

        for guild_id, connection_info in connections.items():
            if connection_info.get("waiting_for_craig", False):
                guild = bot.get_guild(guild_id)
                guild_name = guild.name if guild else f"Guild {guild_id}"
                waiting_connections.append(guild_name)
            elif connection_info.get("detecting", False):
                voice = connection_info.get("voice")
                if voice and voice.channel:
                    active_connections.append(voice.channel.name)

        if waiting_connections:
            status_parts.append(f"\n**크레이그봇 대기 중**: {len(waiting_connections)}개 서버")
            for guild_name in waiting_connections[:5]:
                status_parts.append(f"• {guild_name}")
            if len(waiting_connections) > 5:
                status_parts.append(f"• ... 외 {len(waiting_connections)-5}개")

        if active_connections:
            status_parts.append(f"\n**현재 감지 중**: {len(active_connections)}개 채널")
            for channel_name in active_connections[:5]:
                status_parts.append(f"• {channel_name}")
            if len(active_connections) > 5:
                status_parts.append(f"• ... 외 {len(active_connections)-5}개")

        total_sessions = len(first_speech_sessions)
        total_waiting = len(waiting_connections)
        total_active = len(active_connections)

        status_parts.append(f"\n**상태 요약**")
        status_parts.append(f"• 활성 세션: {total_sessions}개")
        status_parts.append(f"• 대기 상태: {total_waiting}개")
        status_parts.append(f"• 감지 중: {total_active}개")

        final_status = "\n".join(status_parts)

        if len(final_status) > 2000:
            final_status = final_status[:1900] + "\n\n... (내용이 길어 일부 생략됨)"

        await ctx.respond(final_status, ephemeral=True)

    except Exception as e:
        logger.error(f"첫 발화 상태 확인 오류: {e}")
        await ctx.respond(f"상태 확인 중 오류 발생: {e}", ephemeral=True)


@bot.slash_command(description="디버깅 정보를 출력합니다")
async def debug(ctx):
    try:
        debug_info = []
        debug_info.append(f"**Bot 연결 상태:** {'연결됨' if bot.is_ready() else '연결 안됨'}")

        if ctx.guild.id in connections:
            connection_info = connections[ctx.guild.id]
            voice = connection_info.get("voice")

            if connection_info.get("waiting_for_craig", False):
                debug_info.append(f"**현재 상태:** 크레이그봇 대기 중")
                debug_info.append(f"**Voice Client:** 연결 해제됨 (대기 모드)")
            elif voice and voice.channel:
                debug_info.append(f"**현재 상태:** 활성 감지 중")
                debug_info.append(f"**Voice Client:** 연결됨")
                debug_info.append(
                    f"**음성 감지 상태:** {'감지 중' if connection_info.get('detecting', False) else '대기 중'}"
                )
                debug_info.append(f"**채널:** {voice.channel.name}")
                debug_info.append(f"**지연 시간:** {voice.latency * 1000:.1f}ms")
            else:
                debug_info.append(f"**현재 상태:** 알 수 없음")
                debug_info.append(f"**Voice Client:** 연결 안됨")

            if "sink" in connection_info and connection_info["sink"]:
                sink = connection_info["sink"]
                debug_info.append(f"**Sink 상태:** 활성화")
                debug_info.append(f"**감지 쿨다운:** {getattr(sink, 'detection_cooldown', 0)}초")
                debug_info.append(f"**사용자 캐시:** {len(getattr(sink, 'user_cache', {}))}명")
            else:
                debug_info.append(f"**Sink 상태:** 비활성화")
        else:
            debug_info.append(f"**현재 상태:** 초기 상태 (연결 정보 없음)")
            debug_info.append(f"**Voice Client:** 연결 안됨")

        debug_info.append(f"**첫 발화 세션:** {len(first_speech_sessions)}개 활성화")
        debug_info.append(f"**총 연결 정보:** {len(connections)}개 서버")

        waiting_count = sum(1 for info in connections.values() if info.get("waiting_for_craig", False))
        debug_info.append(f"**크레이그봇 대기 중:** {waiting_count}개 서버")

        app_data_path = get_app_data_path()
        debug_info.append(f"**저장 경로:** `{app_data_path}`")

        await ctx.respond("**디버깅 정보**\n" + "\n".join(debug_info), ephemeral=True)

    except Exception as e:
        logger.error(f"디버깅 정보 출력 오류: {e}")
        await ctx.respond(f"디버깅 정보 출력 실패: {e}", ephemeral=True)


@bot.slash_command(description="현재 서버에서 봇을 강제로 음성 채널에서 퇴장시킵니다 (관리자 전용)")
async def force_leave(ctx):
    try:
        if not ctx.author.guild_permissions.administrator:
            await ctx.respond("이 명령어는 관리자만 사용할 수 있습니다.", ephemeral=True)
            return

        await ctx.defer()

        guild_id = ctx.guild.id

        if guild_id not in connections:
            await ctx.followup.send("현재 이 서버에서 연결 정보가 없습니다.", ephemeral=True)
            return

        connection_info = connections[guild_id]
        voice = connection_info.get("voice")
        channel_name = voice.channel.name if voice and voice.channel else "Unknown"
        was_waiting = connection_info.get("waiting_for_craig", False)

        if connection_info.get("detecting", False) and voice:
            voice.stop_recording()
            logger.info("관리자 명령으로 음성 감지 중지")

        sink = connection_info.get("sink")
        if sink:
            sink.user_cache.clear()
            sink.last_detection.clear()
            logger.info("관리자 명령으로 Sink 메모리 정리")

        if voice:
            await voice.disconnect(force=True)
            logger.info(f"관리자 명령으로 음성 채널 연결 해제: {channel_name}")

        del connections[guild_id]

        channels_to_remove = [
            ch_id for ch_id, info in first_speech_sessions.items() if info.get("guild_id") == guild_id
        ]
        for ch_id in channels_to_remove:
            del first_speech_sessions[ch_id]

        status = "대기 모드" if was_waiting else "활성 모드"
        response_text = f"**관리자 강제 퇴장 완료**\n\n"
        response_text += f"**이전 상태:** {status}\n"
        response_text += f"**채널:** {channel_name}\n"
        response_text += f"**실행자:** {ctx.author.mention}\n"
        response_text += f"**완료 작업:** 모든 연결 해제, 메모리 정리, 세션 정보 삭제"

        await ctx.followup.send(response_text)
        logger.info(f"관리자 {ctx.author.name}이 강제 퇴장 명령 실행: {channel_name}")

    except Exception as e:
        logger.error(f"강제 퇴장 명령 실행 중 오류: {e}")
        await ctx.followup.send(f"강제 퇴장 중 오류 발생: {e}", ephemeral=True)


@bot.slash_command(description="모든 서버의 모든 음성 세션을 강제로 중단합니다 (관리자 전용)")
async def stop_all_sessions(ctx):
    try:
        if not ctx.author.guild_permissions.administrator:
            await ctx.respond("이 명령어는 관리자만 사용할 수 있습니다.", ephemeral=True)
            return

        await ctx.defer()

        if not connections and not first_speech_sessions:
            await ctx.followup.send("현재 활성화된 세션이 없습니다.", ephemeral=True)
            return

        stopped_count = 0
        stopped_channels = []
        waiting_count = 0

        for guild_id, connection_info in list(connections.items()):
            try:
                voice = connection_info.get("voice")
                was_waiting = connection_info.get("waiting_for_craig", False)

                if was_waiting:
                    waiting_count += 1
                    guild = bot.get_guild(guild_id)
                    guild_name = guild.name if guild else f"Guild {guild_id}"
                    stopped_channels.append(f"{guild_name} (대기 모드)")
                else:
                    channel_name = voice.channel.name if voice and voice.channel else f"Guild {guild_id}"
                    stopped_channels.append(f"{channel_name} (활성 모드)")

                if connection_info.get("detecting", False) and voice:
                    voice.stop_recording()

                sink = connection_info.get("sink")
                if sink:
                    sink.user_cache.clear()
                    sink.last_detection.clear()

                if voice:
                    await voice.disconnect(force=True)

                stopped_count += 1

            except Exception as e:
                logger.error(f"Guild {guild_id} 세션 중단 중 오류: {e}")

        connections.clear()
        first_speech_sessions.clear()

        response_text = f"**모든 세션 강제 중단 완료**\n\n"
        response_text += f"**총 중단된 세션:** {stopped_count}개\n"
        response_text += f"**활성 세션:** {stopped_count - waiting_count}개\n"
        response_text += f"**대기 세션:** {waiting_count}개\n"
        response_text += f"**실행자:** {ctx.author.mention}\n\n"

        if stopped_channels:
            channels_display = "\n".join(f"• {channel}" for channel in stopped_channels[:10])
            if len(stopped_channels) > 10:
                channels_display += f"\n• ... 외 {len(stopped_channels)-10}개"
            response_text += f"**중단된 세션들:**\n{channels_display}\n\n"

        response_text += f"**완료 작업:** 모든 음성 감지 중지, 모든 메모리 정리, 모든 연결 해제"

        await ctx.followup.send(response_text)
        logger.info(f"관리자 {ctx.author.name}이 모든 세션 중단 명령 실행: {stopped_count}개 세션 중단")

    except Exception as e:
        logger.error(f"모든 세션 중단 명령 실행 중 오류: {e}")
        await ctx.followup.send(f"세션 중단 중 오류 발생: {e}", ephemeral=True)


@bot.slash_command(description="봇의 모든 상태를 초기화합니다 (관리자 전용)")
async def emergency_reset(ctx):
    try:
        if not ctx.author.guild_permissions.administrator:
            await ctx.respond("이 명령어는 관리자만 사용할 수 있습니다.", ephemeral=True)
            return

        await ctx.defer()

        reset_count = 0
        for guild_id, connection_info in list(connections.items()):
            try:
                voice = connection_info.get("voice")
                if voice:
                    try:
                        voice.stop_recording()
                    except:
                        pass
                    try:
                        await voice.disconnect(force=True)
                    except:
                        pass
                reset_count += 1
            except:
                pass

        connections.clear()
        first_speech_sessions.clear()

        import gc

        gc.collect()

        response_text = f"**긴급 재시작 완료**\n\n"
        response_text += f"봇의 모든 상태가 초기화되었습니다.\n"
        response_text += f"크레이그봇 추적이 다시 활성화되었습니다.\n\n"
        response_text += f"**실행자:** {ctx.author.mention}\n"
        response_text += f"**정리된 세션:** {reset_count}개\n"
        response_text += f"**완료 작업:** 모든 연결 강제 해제, 메모리 완전 정리, 가비지 컬렉션, 상태 초기화"

        await ctx.followup.send(response_text)
        logger.info(f"관리자 {ctx.author.name}이 긴급 재시작 명령 실행")
        print(f"긴급 재시작 완료 - 크레이그봇 추적 재활성화")

    except Exception as e:
        logger.error(f"긴급 재시작 명령 실행 중 오류: {e}")
        await ctx.followup.send(f"긴급 재시작 중 오류 발생: {e}", ephemeral=True)


@bot.slash_command(description="크레이그봇 대기 상태인 서버 목록을 확인합니다")
async def waiting_status(ctx):
    try:
        waiting_servers = []

        for guild_id, connection_info in connections.items():
            if connection_info.get("waiting_for_craig", False):
                guild = bot.get_guild(guild_id)
                if guild:
                    waiting_servers.append(guild.name)
                else:
                    waiting_servers.append(f"Unknown Guild ({guild_id})")

        if not waiting_servers:
            await ctx.respond("현재 크레이그봇 대기 중인 서버가 없습니다.", ephemeral=True)
            return

        response_text = f"**크레이그봇 대기 중인 서버 목록** ({len(waiting_servers)}개)\n\n"

        for i, server_name in enumerate(waiting_servers[:20], 1):
            response_text += f"{i}. {server_name}\n"

        if len(waiting_servers) > 20:
            response_text += f"\n... 외 {len(waiting_servers)-20}개 서버"

        response_text += (
            f"\n\n**설명:** 대기 상태는 첫 발화 감지 후 자동 퇴장했거나, 크레이그봇이 퇴장한 후의 상태입니다. "
            f"크레이그봇이 다시 입장하면 자동으로 추적을 재시작합니다."
        )

        await ctx.respond(response_text, ephemeral=True)

    except Exception as e:
        logger.error(f"대기 상태 확인 오류: {e}")
        await ctx.respond(f"대기 상태 확인 중 오류 발생: {e}", ephemeral=True)


def main():
    if not Config.token or Config.token == "YOUR_BOT_TOKEN_HERE":
        print("오류: 봇 토큰이 설정되지 않았습니다!")
        print("환경변수 TOKEN을 설정하거나 Config.token을 수정하세요.")
        return

    try:
        logger.info("Craig Bot 지속 추적 첫 발화 감지 봇 시작 중...")
        print("Craig Bot 지속 추적 첫 발화 감지 봇 시작 중...")
        print("첫 발화 감지 후 자동 퇴장하여 대기 모드로 전환")
        print("크레이그봇 재입장 시 자동으로 추적 재시작")
        bot.run(Config.token)
    except discord.LoginFailure:
        logger.error("잘못된 봇 토큰입니다!")
    except Exception as e:
        logger.error(f"봇 실행 실패: {e}", exc_info=True)


if __name__ == "__main__":
    main()
