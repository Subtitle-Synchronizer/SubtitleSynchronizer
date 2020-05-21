import numpy as np

frame_secs = 0.05


def split_to_frames(data, frame_size):
    n_frames = int(len(data)/frame_size)
    data = data[:(n_frames*frame_size)]
    return np.reshape(data, (n_frames, frame_size))


def shift_frames(frames, delta):
    if delta == 0:
        return frames
    result = frames*0
    if delta > 0:
        result[delta:,:] = frames[:-delta]
    else:
        result[:delta,:] = frames[-delta:,:]
    return result


def expand_to_adjacent(frames, width=1):
    return np.hstack([shift_frames(frames, delta) for delta in range(-width,width+1)])


def rolling_aggregates(frames, width=1, aggregate=np.mean):
    # compute per feature
    result = np.empty(frames.shape)
    shifts = range(-width, width+1)
    for feature_index in range(result.shape[1]):
        feature_col = frames[:,feature_index][:,np.newaxis]
        windows = np.hstack([shift_frames(feature_col, delta) for delta in shifts])
        result[:, feature_index] = aggregate(windows, axis=1)
    return result


def apply_windowing(frames):
    extended = expand_to_adjacent(frames)
    window = np.hanning(extended.shape[1])
    return extended * window[np.newaxis,:]


def compute_spectra(windowed, window_length_secs):
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    frequency = np.arange(spectrum.shape[1]) / window_length_secs
    audible = (frequency > 20) & (frequency < 20000)

    spectrum = spectrum[:, audible]
    return spectrum.astype(np.float32)


def compute_banks(spectra, n_banks):
    bank_size = int(spectra.shape[1] / n_banks)
    banks = []
    for i in range(n_banks):
        begin = i*bank_size
        end = (i+1)*bank_size
        bank = spectra[:,begin:end]
        banks.append(np.log1p(np.sqrt(np.sum(bank**2, axis=1))[:,np.newaxis]))
    return np.hstack(banks)


def split_to_chunks(n, chunk_size):
    i_begin = 0
    while i_begin < n:
        i_end = min(n, i_begin+chunk_size)
        yield(slice(i_begin, i_end))
        i_begin = i_end


def maybe_parallel_map(f, data, parallelism=1):
    if parallelism > 1:
        from multiprocessing import Pool
        try:
            pool = Pool(parallelism)
            return pool.map(f, data)
        finally:
            pool.close()
    else:
        return [f(c) for c in data]


def compute_chunk_features(sound_data_chunk, data_range, frame_size, frame_secs):
    training_x = split_to_frames(sound_data_chunk, frame_size)
    spectra = compute_spectra(apply_windowing(training_x)/float(data_range), 3*frame_secs)
    return compute_banks(spectra, n_banks=50)


def _compute_chunk_features_star(args):
    return compute_chunk_features(*args)


def compute(sound_data, subvec, parallelism=3):
    samples, sample_rate, data_range = sound_data
    frame_size = int(frame_secs*sample_rate)

    # compute features in chunks
    chunk_size_secs = 120
    chunk_size = int(chunk_size_secs / frame_secs)
    chunks = list(split_to_chunks(len(samples), chunk_size))

    def compute_chunk_labels(chunk):
        return np.round(np.mean(split_to_frames(subvec[chunk], frame_size), axis=1)).astype(np.float32)

    data_chunks = [(samples[c], data_range, frame_size, frame_secs) for c in chunks]
    all_x = np.vstack(maybe_parallel_map(_compute_chunk_features_star, data_chunks, parallelism))
    all_y = np.hstack([compute_chunk_labels(c) for c in chunks])
    return all_x, all_y


def compute_train_features(sound_data, parallelism=3):
    samples, sample_rate, data_range = sound_data
    frame_size = int(frame_secs*sample_rate)

    # compute features in chunks
    chunk_size_secs = 120
    chunk_size = int(chunk_size_secs / frame_secs)
    chunks = list(split_to_chunks(len(samples), chunk_size))

    data_chunks = [(samples[c], data_range, frame_size, frame_secs) for c in chunks]
    all_x = np.vstack(maybe_parallel_map(_compute_chunk_features_star, data_chunks, parallelism))
    return all_x


def weight_by_group(group_labels):
    weights = np.zeros(len(group_labels))

    for g in np.unique(group_labels):
        part = group_labels == g
        w = 1.0 / np.sum(part)
        weights[part] = w

    return weights / np.mean(weights)
