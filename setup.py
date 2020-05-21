from setuptools import setup
from os import path

# Get the long description from the DESCRIPTION file
with open('DESCRIPTION.rst') as f:
    long_description = f.read()

package_name = 'SubtitleSynchronizer'

setup(
    name=package_name,
    version='0.0.3',
    description='Automatically synchronize subtitles with video files',
    long_description=long_description,
    url='https://github.com/Subtitle-Synchronizer/' + package_name,
    author='Abhishek Khandelwal',

    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Environment :: Console',
        'Operating System :: POSIX',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Multimedia',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3'
    ],
    keywords='subtitles syncrhonization srt ffmpeg spleeter',
    packages=['SubtitleSynchronizer'],

    # distribute the trained model file in the package
    package_data={ package_name: ['../trained.model.spleeter.bin'] },

    # define command line entry point
    entry_points = {
        'console_scripts': [
            "%s=%s.main:cli_packaged" % (package_name, package_name)
        ]
    },

    install_requires=['numpy', 'pysoundfile'],

    extras_require={
        # dev requirements are needed for training the model
        'dev': ['pandas', 'sklearn'],
        'test': ['nose']
    }
)
