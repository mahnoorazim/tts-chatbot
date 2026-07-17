"""

1. Load Kokoro once.
2. Ask you to enter text in the terminal.
3. Convert the text into speech.
4. Print the generated phonemes.
5. Save the speech as a WAV file.
6. Continue accepting text until you type quit, exit, or q.
"""

from datetime import datetime
from pathlib import Path
import time
import soundfile as sf
import torch
from kokoro import KPipeline



LANGUAGE_CODE = "a" # American English
VOICE = "af_heart" 
SPEED = 1.0
DEVICE = "cpu"
SAMPLE_RATE = 24_000

OUTPUT_DIRECTORY = Path("outputs/interactive")
EXIT_COMMANDS = {"Q", "q"}


def create_output_path() -> Path:

    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")
    return OUTPUT_DIRECTORY / f"speech_{timestamp}.wav"


def generate_speech(
    pipeline: KPipeline,
    text: str,
) -> tuple[Path, float, float, float, int]:
    """
    Generate and save speech for one piece of text.

    Parameters
    ----------
    pipeline:
        The already-loaded Kokoro pipeline.

    text:
        The sentence or paragraph entered by the user.

    Returns
    -------
    A tuple containing:

    1. The saved WAV file path
    2. Speech generation time in seconds
    3. Generated audio duration in seconds
    4. Real-time factor
    5. Number of generated segments
    """

    generation_start = time.perf_counter()

    generator = pipeline(
        text,
        voice=VOICE,
        speed=SPEED,
        split_pattern=r"\n+",
    )

    audio_segments: list[torch.Tensor] = []

    segment_count = 0

    for segment_index, (
        graphemes,
        phonemes,
        audio,
    ) in enumerate(generator, start=1):

        segment_count += 1

        print()
        print(f"Segment {segment_index}")
        print("-" * 40)

        print("Graphemes:")
        print(graphemes)

        print()
        print("Phonemes:")
        print(phonemes)

        if audio is None:
            print()
            print("Warning: this segment contained no audio.")
            continue

        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)

  
        prepared_audio = (
            audio
            .detach()
            .cpu()
            .flatten()
        )

        audio_segments.append(prepared_audio)

    if not audio_segments:
        raise RuntimeError(
            "Finished without producing any valid audio."
        )

    complete_audio_tensor = torch.cat(
        audio_segments,
        dim=0,
    )

    complete_audio = complete_audio_tensor.numpy()

    output_path = create_output_path()

    sf.write(
        file=output_path,
        data=complete_audio,
        samplerate=SAMPLE_RATE,
        subtype="PCM_16",
    )

    generation_seconds = (
        time.perf_counter() - generation_start
    )

    audio_duration_seconds = (
        len(complete_audio) / SAMPLE_RATE
    )

    real_time_factor = (
        generation_seconds / audio_duration_seconds
    )

    return (
        output_path,
        generation_seconds,
        audio_duration_seconds,
        real_time_factor,
        segment_count,
    )


def main() -> None:

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Kokoro...")


    pipeline = KPipeline(
        lang_code=LANGUAGE_CODE,
        device=DEVICE,
    )

    print()
    print("Kokoro is ready.")
    print(f"Voice: {VOICE}")
    print(f"Device: {DEVICE}")
    print(f"Output folder: {OUTPUT_DIRECTORY}")
    print()
    print('Type text to generate speech.')
    print('Type "Q" or "q" to exit the program.')

    while True:
        print()

        try:
            user_text = input("Enter text: ").strip()

        except (KeyboardInterrupt, EOFError):
            print()
            print("Closing interactive TTS.")
            break

        if not user_text:
            print("No text entered! Please enter a sentence.")
            continue

        if user_text in EXIT_COMMANDS:
            print("Closing interactive TTS.")
            break

        print()
        print("Generating speech...")

        try:
            (
                output_path,
                generation_seconds,
                audio_duration_seconds,
                real_time_factor,
                segment_count,
            ) = generate_speech(
                pipeline=pipeline,
                text=user_text,
            )

        except Exception as error:
            print()
            print("Speech generation failed.")
            print(f"Error: {error}")
            continue

        print()
        print("Speech generated successfully.")
        print(f"Saved to: {output_path}")
        print(f"Segments generated: {segment_count}")
        print(
            f"Generation time: "
            f"{generation_seconds:.2f} seconds"
        )
        print(
            f"Audio duration: "
            f"{audio_duration_seconds:.2f} seconds"
        )
        print(
            f"Real-time factor: "
            f"{real_time_factor:.3f}"
        )
        if real_time_factor < 1.0:
            print("Generation was faster than real-time.")
        else:
            print("Generation was slower than real-time.")


if __name__ == "__main__":
    main()