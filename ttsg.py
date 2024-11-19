from gtts import gTTS
from langdetect import detect
from pydub import AudioSegment
from pydub.playback import play

def txt2spch(text, lang):
    
    if not text.strip():
        print("No valid text detected for speech synthesis. Skipping...")
        return
    
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        audio = AudioSegment.from_mp3("output.mp3")
        play(audio)
    except AssertionError:
        print("No text to send to TTS API. Skipping this input...")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
#txt2spch("Bonjour ! Je suis votre assistant vocal, voici ce qui est écrit en direction de votre caméra traduit dans votre langue", 'fr')
#txt2spch("Hello! I am your voice assistant, here is what is written in the direction of your camera translated into your language", 'en')
