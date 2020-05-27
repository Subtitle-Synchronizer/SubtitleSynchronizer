Subtitle Synchronizer - with the application of AI and Probability concepts

Subtitle Synchronization is an interesting problem to solve with the application of 
Machine Learning and Probability.

There are already quite a few python libraries that deal with this problem and this
library is inspired from a famous library called <a href="https://github.com/oseiskar/autosubsync">'autosubsync'</a>.

Subtitle Synchronization has two major processing steps:

--> Vocal section identification
    This processing is associated with identification of vocal section present in video file. This library
uses <a href="https://github.com/deezer/spleeter">Spleeter</a>, open source audio separation library for this purpose and then uses audio processing
technique to build the vocal classification model.

This library differs from 'autosubsync' on this step. 'autosubsync' uses the labeling of vocal section
with reference from SRT file. This has some inherent limitations with regard to purpose of SRT file.
We would like to address this limitation with the application of Spleeter and other vocal processing
techniques.

We achieve very high accuracy with the current method and we believe this improved accuracy
would have a positive impact on the overall accuracy of subtitle synchronization.

--> Subtitle correction
    Subtitle timing correction based on 'Vocal Classification' results is second and final process involved in this activity.
Here, we reuse the processing of 'autosubsync' library completely. Author of autosubsync has done an
outstanding job in applying 'bernoulli equation' in solving this step. It was a
very good learning for us to understand the application. 

Pls refer this <a href="https://medium.com/@vvk.victory/subtitle-synchronization-ai-probabilitistc-approach-to-the-rescue-e59e166c5f25?postPublishedType=initial">blog</a> for the detailed information about the processing involved in this library. 


## Installation

### macOS / OSX
Prerequisites: Install [Homebrew](https://brew.sh/) and [pip](https://stackoverflow.com/questions/17271319/how-do-i-install-pip-on-macos-or-os-x). Then install FFmpeg, [spleeter] (https://github.com/deezer/spleeter) and this package

```
brew install ffmpeg
pip install spleeter
pip install subtitlesynchronizer
```

### Linux (Debian & Ubuntu)

Make sure you have Pip, e.g., `sudo apt-get install python-pip`.
Then install [FFmpeg](https://www.ffmpeg.org/), [spleeter] (https://github.com/deezer/spleeter) and this package
```
sudo apt-get install ffmpeg
sudo pip install spleeter
sudo pip install subtitlesynchronizer
```

Note: If you are running Ubuntu 14 (but not 12 and 16, which are fine), you'll need to [jump some more hoops to install FFmpeg](https://www.faqforge.com/linux/how-to-install-ffmpeg-on-ubuntu-14-04/).

## Usage

```
subtitlesynchronizer [input movie] [input subtitles] [output subs]

# for example
subtitlesynchronizer plan-9-from-outer-space.avi \
  plan-9-out-of-sync-subs.srt \
  plan-9-subtitles-synced.srt
```
See `subtitlesynchronizer --help` for more details.


## Development

### Data Generation

 1. For Vocal audio files, we referred 'LibriSpeech' dataset. Refer this [link] (http://www.openslr.org/12)
 2. For non-vocal (background noise/sound), we referred urban dataset
 3. For non-vocal (mute sound), we used spleeter to extract mute sections from given autio file

### Training the model

 1. Collect about 1000 vocal audio files and put them
    in a folder called `audioFiles/1`
 2. Collect about 1000 non-vocal audio files and put them
    in a folder called `audioFiles/0`
 3. Run (and see) `train_and_test.sh`. This
    * creates `subtitlesynchronizer.trained.model.bin`
    * runs cross-validation

### Synchronization (predict)

Assumes trained model is available as `subtitlesynchronizer.trained.model.binn` under `SubtitleSynchronizer` folder

    python3 SubtitleSynchronizer/main.py input-video-file input-subs.srt synced-subs.srt

### Build and distribution

 * Create virtualenv: `python3 -m venv venvs/test-python3`
 * Activate venv: `source venvs/test-python3/bin/activate`
 * `pip install -e .`
 * `pip install wheel`
 * `python setup.py bdist_wheel`
