HikimaEngine is an LLM project with a mission to provide equitable access to quality education in linguistically diverse regions focused on major languages in Nigeria. The LLM offers a comprehensive suite of tools, such as translation/transcription of educational materials and tailoring learning experience.

Installation 🦉
These instructions should work for most cases, but I heard of some instances where espeak behaves weird, which are sometimes resolved after a re-install and sometimes not. Also, M1 and M2 MacBooks require a very different installation process, with which I am unfortunately not familiar.

Basic Requirements
To install this toolkit, clone it onto the machine you want to use it on (should have at least one cuda enabled GPU if you intend to train models on that machine. For inference, you don't need a GPU). Navigate to the directory you have cloned. We recommend creating and activating a virtual environment to install the basic requirements into. The commands below summarize everything you need to do under Linux. If you are running Windows, the second line needs to be changed, please have a look at the venv documentation.

## Prerequisites

- Python 3.8 or higher
- Flask
- [FFmpeg](https://ffmpeg.org/download.html)

## Technologies Used

- [Whisper ASR](https://www.openai.com/research/whisper/): Used to transcribe the audio from the video file.
- [Spacy](https://spacy.io/): Used for natural language processing tasks, such as tokenization and syllable counting.
- [PyDub](http://pydub.com/): Used for manipulating audio files.
- [MoviePy](https://zulko.github.io/moviepy/): Used for extracting the audio from the video file.


## Installation

1. Clone this repository:
   ```
   git clone https://github.com/DiniMuhd7/hikima-engine.git
   ```
2. Create a virtual environment and activate it in the project directory
   ```
   python -m venv <path_to_where_you_want_your_env_to_be>
   source <path_to_where_you_want_your_env_to_be>/bin/activate
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
Run the second line everytime you start using the tool again to activate the virtual environment again, if you e.g. logged out in the meantime. To make use of a GPU, you don't need to do anything else on a Linux machine. On a Windows machine, have a look at the official PyTorch website for the install-command that enables GPU support.

## [optional] eSpeak-NG
eSpeak-NG is requirement, that handles lots of special cases in many languages, so it's good to have.

On most Linux environments it will be installed already, and if it is not, and you have the sufficient rights, you can install it by simply running
```
apt-get install espeak-ng
```

For Windows, they provide a convenient .msi installer file on their GitHub release page. After installation on non-linux systems, you'll also need to tell the phonemizer library where to find your espeak installation by setting the PHONEMIZER_ESPEAK_LIBRARY environment variable, which is discussed in this issue.

For Mac it's unfortunately a lot more complicated. Thanks to Sang Hyun Park, here is a guide for installing it on Mac: For M1 Macs, the most convenient method to install espeak-ng onto your system is via a MacPorts port of espeak-ng. MacPorts itself can be installed from the MacPorts website, which also requires Apple's XCode. Once XCode and MacPorts have been installed, you can install the port of espeak-ng via
```
sudo port install espeak-ng
```

As stated in the Windows install instructions, the espeak-ng installation will need to be set as a variable for the phonemizer library. The environment variable is PHONEMIZER_ESPEAK_LIBRARY as given in the GitHub thread linked above. However, the espeak-ng installation file you need to set this variable to is a .dylib file rather than a .dll file on Mac. In order to locate the espeak-ng library file, you can run port contents espeak-ng. The specific file you are looking for is named libespeak-ng.dylib.




