"""


1. Load the pretrained Kokoro-82M model locally.
2. Convert one fixed English sentence into speech.
3. Print the grapheme and phoneme representations.
4. Combine all generated audio segments.
5. Save the final speech as a WAV file.

"""

from pathlib import Path
import soundfile as sf
import torch
from kokoro import KPipeline



LANGUAGE_CODE = "a" # American English
VOICE = "af_heart" 
SPEED = 1.0
DEVICE = "cpu"
SAMPLE_RATE = 24_000
TEST_TEXT = (
    "Hello! This is the first test of our "
    "chatbot text-to-speech system."
)
OUTPUT_PATH = Path(
    "outputs/baseline/samples/00_test.wav"
)


def main() -> None:


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Kokoro text-to-speech...")

    
    pipeline = KPipeline(
        lang_code=LANGUAGE_CODE,
        device=DEVICE,
    )
    print("Kokoro pipeline loaded successfully.")
    #Print settings
    print()
    print("Configuration-")
   
    print(f"Language code: {LANGUAGE_CODE}")
    print(f"Voice: {VOICE}")
    print(f"Speed: {SPEED}")
    print(f"Device: {DEVICE}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")

    # Print text being converted to speech
    print()
    print("Input text")
    print("----------")
    print(TEST_TEXT)

    # Send the text to Kokoro
   
    generator = pipeline(
        TEST_TEXT,
        voice=VOICE,
        speed=SPEED,
        split_pattern=r"\n+",
    )

    # List that will store audio segments as tensors with waveforms.
    
    audio_segments: list[torch.Tensor] = []

    
    for segment_index, (
        graphemes,
        phonemes,
        audio,
    ) in enumerate(generator):

        print()
        print(f"Generated segment {segment_index}")

        print("Graphemes:")
        print(graphemes)

        print("Phonemes:")
        print(phonemes)

        if audio is None:
            print("This segment does not contain audio.")
            continue

        
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)

        cleaned_audio = (
            audio
            .detach()
            .cpu()
            .flatten()
        )
        audio_segments.append(cleaned_audio)

    if not audio_segments:
        raise RuntimeError(
            "Finished without returning any valid audio."
        )

    #Join tensors alone time dimension to create single audio tensor
    complete_audio_tensor = torch.cat(
        audio_segments,
        dim=0,
    )

    complete_audio = complete_audio_tensor.numpy()

    # Save the audio as WAV file.
    
    sf.write(
        file=OUTPUT_PATH,
        data=complete_audio,
        samplerate=SAMPLE_RATE,
        subtype="PCM_16",
    )

    # Pirnt final results
    print()
    print("Initial test completed successfully.")
    print(f"Saved audio to: {OUTPUT_PATH}")
    print(
        f"Audio samples generated: "
        f"{len(complete_audio):,}"
    )
    
    print(f"Sample rate: {SAMPLE_RATE} Hz")

if __name__ == "__main__":
    main()