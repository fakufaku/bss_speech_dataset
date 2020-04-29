BSS Dataset
===========

Dataset of reverberant speech mixtures created from CMU Arctic samples and pyroomacoustics.

Create Dataset
--------------

The dependencies are `numpy`, `scipy`, and `pyroomacoustics`.
Anaconda is an easy way to install them

    conda env create -f environment.yml
    conda activate bss_speech_dataset

Running the following script will create the dataset and store it in the `data` folder.

    python ./make_dataset.py config.json

The content of the dataset is described in the file `data/metadata.json` with following structure

    channelsX: [X is the number of channels]
      - [list of 100 rooms]:
        - room_id: the id of the room [0-99]
        - n_channels: the number of channels used [X]
        - room_params: the parameters used to simulate the room
        - mix_filename: the names of the files containing the mixture signals
        - src_filenames: the names of the files containing isolated sources
        - anechoic_filenames: the names of the files containing the isolated
          sources without reverberation, but with the correct time of arrivals,
          this is useful for evaluating dereverberation algorithms
        - rir_filenames: the names of the files containing the room impulse responses

File naming scheme

    channelsX_roomY_mix.wav
      The file that contains an X-channels mixture of X sources in room Y

    channelsX_roomY_micZ.wav
      The file that contains X-channels with each isolated source in one of the channels
      all recorded by microphone with index Z in room Y

    channelsX_roomY_micZ_anechoic.wav
      The file that contains X-channels with each non-reverberant isolated source in one 
      of the channels all recorded by microphone with index Z in room Y

    rir_channelsX_roomY_micZ_srcT.wav
      The file that contains the impulse response between source T and microphone Z in
      room Y for the X-channels mixture
