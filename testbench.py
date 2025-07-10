import subprocess

# Should be a valid WAV file created via ffmpeg
subprocess.run(["aplay", "sounds/buy.wav"])