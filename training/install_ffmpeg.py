import subprocess
import sys

from huggingface_hub.utils import capture_output


def install_ffmpeg():
    print("Starting ffmpeg installation...")

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'setuptools'])

    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ffmpeg-python'])
        print("FFmpeg installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg installation failed: {str(e)}")

    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz"
        ])

        subprocess.check_call([
            "tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"
        ])

        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True
        )
        ffmpeg_path = result.stdout.strip()

        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("Installed static FFmpeg binary successfully")
    except Exception as e:
        print(f"Failed to install static FFmpeg: {e}")

    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("FFmpeg version:", result.stdout.strip())
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Ffmpeg installation failed")
        return False

