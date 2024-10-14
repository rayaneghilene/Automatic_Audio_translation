# Automatic Audio translation

This repository contains the code for the automatic audio2audio translation. We use whisper for transcription, BART for translation, and speechTT5 for audio playback generation. 

![Pipeline](https://github.com/user-attachments/assets/d566590c-f649-49d8-b818-cb8777c3e74d)

This is an early version of the code, a html/css UI, and fastAPI is provided with the code. 


## Usage

Clone the repo using the following command:
```ruby
git clone https://github.com/rayaneghilene/Automatic_Audio_translation.git
cd Audio2Audio_translation
```

We recommend creating a virtual environment (optional)
```bash
python3 -m venv myenv
source myenv/bin/activate 
```

To install the requirements run the following command: 
```bash
pip install -r requirements.txt
```

Use the following command to run the script, this will prompt an html/css UI where you can drag and drop your audio file. Once the processing is done, an output pile in english will be automatically downloaded to your machine

```ruby
fastapi dev main.py
```

The UI can be accessible at http://127.0.0.1:8000 
Here's a preview:



https://github.com/user-attachments/assets/ebe8f0b8-5b64-4788-b9e3-02e7ffadf956


## Futur improvements
1. Real-time audio processing (double buffering, streaming...)
2. Voice cloning
3. chrome extension?

## Contributing
We welcome contributions from the community to enhance work. If you have ideas for features, improvements, or bug fixes, please submit a pull request or open an issue on GitHub.

## Contact
Feel free to reach out about any questions/suggestions at rayane.ghilene@ensea.fr
