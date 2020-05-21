Automatically synchronize SRT subtitles with audio.

Requires:

ffmpeg (``pip install ffmepg``)

spleeter (``pip install spleeter``)
::

  SubtitleSynchronizer [input movie] [input subtitles] [output subs]

  # for example
  SubtitleSynchronizer plan-9-from-outer-space.avi \
    plan-9-out-of-sync-subs.srt \
    plan-9-subtitles-synced.srt

See ``SubtitleSynchronizer --help`` for more details.
