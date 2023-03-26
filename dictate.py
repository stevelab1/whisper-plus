import multiprocessing
import signal
import io
import os
import tempfile
import threading
import queue
import click
import torch
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import whisper
from datetime import datetime

multiprocessing.set_start_method('fork')  # added line


stop_threads = False


def signal_handler(sig, frame):
    global stop_threads
    stop_threads = True
    print("\nTerminating... Merging transcripts...")
    merge_transcripts()
    print("Merged transcripts successfully.")
    os._exit(0)


@click.command()
# more advanced models may produce better results, but are much slower
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "small", "medium", "large"]))
@click.option("--english", default=False, help="Whether to use English model", is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic engergy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--save_file", default=False, help="Flag to save file", is_flag=True, type=bool)
def main(model, english, verbose, energy, pause, dynamic_energy, save_file):
    signal.signal(signal.SIGINT, signal_handler)
    try:

        # Create the recordings directory if it does not exist
        if save_file:
            if not os.path.exists("recordings"):
                os.makedirs("recordings")

        # Load the audio model and start the recording and transcription threads
        audio_model = whisper.load_model(model)
        audio_queue = queue.Queue()
        result_queue = queue.Queue()
        threading.Thread(target=record_audio,
                         args=(audio_queue, energy, pause, dynamic_energy, save_file)).start()
        threading.Thread(target=transcribe_forever,
                         args=(audio_queue, result_queue, audio_model, english, verbose)).start()

        while True:
            print(result_queue.get())

    except KeyboardInterrupt:
        print("\nTerminating... Merging transcripts...")
        merge_transcripts()
        print("Merged transcripts successfully.")


def record_audio(audio_queue, energy, pause, dynamic_energy, save_file):
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    # Current OpenAI Whisper model trained on audio with 16kHz sample rate — openai.com/papers/whisper.pdf
    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        while True:
            audio = r.listen(source)
            output_filename = generate_output_filename()
            with open(output_filename, "wb") as f:
                f.write(audio.get_wav_data())
            torch_audio = torch.from_numpy(np.frombuffer(
                audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_queue.put_nowait(torch_audio)


def transcribe_forever(audio_queue, result_queue, audio_model, english, verbose):
    while True:
        audio_data = audio_queue.get()
        if english:
            result = audio_model.transcribe(
                audio_data, language='english', fp16=False)  # specify fp16=False
        else:
            result = audio_model.transcribe(audio_data)

        output_filename = generate_output_filename().replace(".wav", ".txt")
        with open(output_filename, "w") as f:
            f.write(result["text"])

        if not verbose:
            predicted_text = result["text"]
            result_queue.put_nowait(predicted_text)
        else:
            result_queue.put_nowait(result)


def merge_transcripts():
    transcript_files = sorted(
        [f for f in os.listdir("recordings") if f.endswith(".txt")])
    merged_transcript_filename = os.path.join(
        "recordings", "merged_transcript.txt")

    with open(merged_transcript_filename, "w") as merged_file:
        for txt_file in transcript_files:
            with open(os.path.join("recordings", txt_file), "r") as f:
                content = f.read()
                merged_file.write(content + "\n")


def generate_output_filename():
    if not os.path.exists("recordings"):
        os.makedirs("recordings")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join("recordings", f"recording_{timestamp}.wav")


if __name__ == '__main__':
    main()
