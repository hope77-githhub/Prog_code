from cx_Freeze import setup, Executable
import sys

# 빌드 옵션 설정
build_exe_options = {
    # 스크립트에서 사용하는 패키지 목록
    "packages": ["os", "sys", "time", "pyaudio", "webrtcvad"],
    # 포함할 추가 파일(필요시)
    "include_files": []
}

# 콘솔 애플리케이션으로 빌드
base = None
if sys.platform == "win32":
    base = None  # 콘솔 창 표시 (GUI 앱이 아님)

# setup 호출
setup(
    name="AudioRecorderWithVAD",
    version="1.0",
    description="Audio recording script with WebRTC VAD padding",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            script="main.py",  # 실제 스크립트 파일명으로 변경
            base=base,
            targetName="audio_recorder.exe"
        )
    ]
)
