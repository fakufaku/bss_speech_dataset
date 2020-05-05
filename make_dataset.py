# Copyright (c) 2019 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This file contains the code to run the systematic simulation for evaluation
of overiva and other algorithms.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import pyroomacoustics as pra
# Get the data if needed
from generate_samples import generate_samples, sampling

# find the absolute path to this file
base_dir = os.path.abspath(os.path.split(__file__)[0])


def create_mixture(args):

    import numpy as np
    from scipy.io import wavfile
    from room_builder import random_room_builder

    # set MKL to only use one thread if present
    try:
        import mkl

        mkl.set_num_threads(1)
    except ImportError:
        pass

    n_mics, room_id, room_seed, wav_files, transcripts, config_filename = args

    # number of sources is implicit from number of source signals provided
    n_sources = len(wav_files)

    with open(config_filename, "r") as f:
        config = json.load(f)

    output_dir = Path(config["dir"])

    # Get a random room with same number of mic and sources
    room, room_params = random_room_builder(
        wav_files, n_mics, seed=room_seed, **config["room_params"]
    )
    premix = room.simulate(return_premix=True)

    # normalize power at reference microphone
    power_norm = np.std(premix[:, config["ref_mic"], :], axis=1)
    premix /= power_norm[:, None, None]

    # mix the signal
    mix = premix.sum(axis=0)

    # Now generate the anechoic version
    anechoic_room = pra.ShoeBox(room.shoebox_dim, fs=room.fs, max_order=0)
    anechoic_room.add_microphone_array(room.mic_array.R)
    for source in room.sources:
        anechoic_room.add_source(source.position, signal=source.signal)
    anechoic_premix = (
        anechoic_room.simulate(return_premix=True) / power_norm[:, None, None]
    )

    # Scale so that all signals fit in the range of int16
    scaling = 2 ** 15 / np.max(
        [np.abs(mix).max(), np.abs(premix).max(), np.abs(anechoic_premix).max()]
    )
    mix = (mix * scaling).astype(np.int16)
    premix = (premix * scaling).astype(np.int16)
    anechoic_premix = (anechoic_premix * scaling).astype(np.int16)

    # Save all the signals
    mix_filename = f"channels{n_mics}_room{room_id}_mix.wav"
    wavfile.write(output_dir / mix_filename, config["room_params"]["fs"], mix.T)

    src_filenames = []
    for m in range(n_mics):
        src_filenames.append(f"channels{n_mics}_room{room_id}_mic{m}.wav")
        wavfile.write(
            output_dir / src_filenames[-1],
            config["room_params"]["fs"],
            premix[:, m, :].T,
        )

    anechoic_filenames = []
    for m in range(n_mics):
        anechoic_filenames.append(f"channels{n_mics}_room{room_id}_mic{m}_anechoic.wav")
        wavfile.write(
            output_dir / anechoic_filenames[-1],
            config["room_params"]["fs"],
            anechoic_premix[:, m, :].T,
        )

    # save the room impulse response
    rir_filenames = []
    for m in range(n_mics):
        for s in range(n_sources):
            rir_filenames.append(
                f"rir_channels{n_mics}_room{room_id}_mic{m}_src{s}.wav"
            )
            wavfile.write(
                output_dir / rir_filenames[-1],
                config["room_params"]["fs"],
                (room.rir[m][s] * 4 * np.pi * (2 ** 15)).astype(np.int16),
            )

    results = {
        "room_id": room_id,
        "n_channels": n_mics,
        "room_params": room_params,
        "mix_filename": mix_filename,
        "src_filenames": src_filenames,
        "anechoic_filenames": anechoic_filenames,
        "rir_filenames": rir_filenames,
        "transcripts": transcripts,
    }

    return results


def generate_arguments(config_filename):
    """ This will generate the list of arguments to run simulation for """

    with open(config_filename, "r") as f:
        config = json.load(f)

    rng_state = np.random.get_state()
    np.random.seed(config["seed"])

    # Maximum total number of sources
    n_sources = np.max(config["n_channels_list"])

    # First we randomly select all the speech samples
    gen_files_seed = int(np.random.randint(2 ** 32, dtype=np.uint32))
    all_wav_files = sampling(
        config["n_repeat"],
        n_sources,
        config["samples_list"],
        gender_balanced=True,
        seed=gen_files_seed,
    )

    # now get the transcripts
    with open(config["samples_list"]) as f:
        dataset_metadata = json.load(f)

    all_transcripts = []
    for subset in all_wav_files:
        transcripts = []
        for fn in subset:
            transcripts.append(dataset_metadata["transcripts"][Path(fn).name])
        all_transcripts.append(transcripts)

    # Pick the seeds to reproducibly build a bunch of random rooms
    room_seeds = np.random.randint(
        2 ** 32, size=config["n_repeat"], dtype=np.uint32
    ).tolist()

    args = []
    for n_channels in config["n_channels_list"]:
        for room_id, (wav_files, transcripts, room_seed) in enumerate(
            zip(all_wav_files, all_transcripts, room_seeds)
        ):

            # add the new combination to the list
            args.append(
                [
                    n_channels,
                    room_id,
                    room_seed,
                    wav_files[:n_channels],
                    transcripts[:n_channels],
                    config_filename,
                ]
            )

    np.random.set_state(rng_state)

    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generates mixture samples for blind source separation"
    )
    parser.add_argument("config", type=str, help="Configuration file")
    parser.add_argument(
        "--cmudir", type=str, help="Directory of CMU ARCTIC corpus, if available"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # construct the samples dataset if necessary
    samples_metadata_file = Path(config["samples_list"])
    samples_dir = samples_metadata_file.parent
    if samples_dir.exists():

        assert samples_metadata_file.exists(), (
            "Samples metadata file does not exist. "
            "Specify a different folder or delete the folder."
        )

        with open(samples_metadata_file) as f:
            samples_metadata = json.load(f)

        assert (
            config["samples_generation_args"] == samples_metadata["generation_args"]
        ), (
            "The samples were not generated by the same argument. "
            "Specify a different folder or clean-up first"
        )

    else:

        print("Generating the samples first...")
        generate_samples(
            cmudir=args.cmudir,
            output_dir=samples_dir,
            **config["samples_generation_args"],
        )

    # create output directory
    if not os.path.exists(config["dir"]):
        os.mkdir(config["dir"])

    all_args = generate_arguments(args.config)

    # Create all the rooms in parallel
    """
    results = []
    for arg in all_args:
        results.append(create_mixture(arg))
    """
    with Pool() as pool:
        results = pool.map(create_mixture, all_args)

    # Rearange the results in a better structure before saving
    metadata = {}
    for r in results:

        room_id = r["room_id"]
        n_channels = r["n_channels"]
        label = f"{n_channels}_channels"

        if label not in metadata:
            metadata[label] = [None for i in range(config["n_repeat"])]

        metadata[label][room_id] = r

    with open(Path(config["dir"]) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
