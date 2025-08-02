import math
import numpy as np

def coeff_exponential(ms, sample_rate=44100.0):
    time_in_samples = ms * sample_rate / 1000.0
    return math.exp(-1.0 / time_in_samples)

class LevelDetector:
    class DetectionType:
        RMS = 'rms'
        PEAK = 'peak'

    def __init__(self):
        self.detection_type = self.DetectionType.RMS
        self.fs = 44100.0
        self.attack_coeff = 0.0
        self.release_coeff = 0.0
        self.prev_output = []

    def init(self, fs, attack_ms, release_ms, num_channels):
        self.fs = fs
        self.set_attack_time(attack_ms)
        self.set_release_time(release_ms)
        self.prev_output = [0.0] * num_channels

    def set_attack_time(self, attack_ms):
        self.attack_coeff = coeff_exponential(attack_ms, self.fs)

    def set_release_time(self, release_ms):
        self.release_coeff = coeff_exponential(release_ms, self.fs)

    def set_fs(self, fs):
        self.fs = fs

    def set_detection_type(self, dtype):
        self.detection_type = dtype

    def reset(self):
        self.prev_output = [0.0] * len(self.prev_output)

    def process_sample(self, sample, channel):
        if self.detection_type == self.DetectionType.RMS:
            processed = sample * sample
        else:
            processed = abs(sample)

        coeff = self.attack_coeff if processed > self.prev_output[channel] else self.release_coeff
        output = (1.0 - coeff) * processed + coeff * self.prev_output[channel]
        self.prev_output[channel] = output
        return output

class FeedForwardCompressor:
    def __init__(self):
        self.threshold_db = 0.0
        self.ratio = 1.0
        self.attack_time = 1.0
        self.release_time = 100.0
        self.fs = 44100.0

        self.threshold = 1.0
        self.threshold_inv = 1.0
        self.ratio_inv = 1.0
        self.precomputed_exp = 0.0

        self.detector = LevelDetector()
        self.prev_gain_l = []
        self.prev_gain_r = []
        self.count_l = 0
        self.count_r = 0
        self.sum_gain_l = 0.0
        self.sum_gain_r = 0.0
        self.prev_gain_size = 0
        self.inv_prev_gain_size = 1.0

    def init(self, fs, num_channels):
        self.fs = fs
        self.detector.init(fs, self.attack_time, self.release_time, num_channels)
        self.set_threshold(self.threshold_db)
        self.set_ratio(self.ratio)
        self.set_gain_interp_length(128)
        self.set_detection_type(LevelDetector.DetectionType.RMS)

    def set_threshold(self, threshold_db):
        self.threshold_db = threshold_db
        self.threshold = 10.0 ** (threshold_db / 20.0)
        self.threshold_inv = 1.0 / self.threshold

    def set_ratio(self, ratio):
        self.ratio = ratio
        self.ratio_inv = 1.0 / ratio
        self.precomputed_exp = self.ratio_inv - 1.0

    def set_attack_time(self, attack_time):
        self.attack_time = attack_time
        self.detector.set_attack_time(attack_time)

    def set_release_time(self, release_time):
        self.release_time = release_time
        self.detector.set_release_time(release_time)

    def set_gain_interp_length(self, length):
        self.prev_gain_size = length
        self.inv_prev_gain_size = 1.0 / length

        self.prev_gain_l = [1.0] * length
        self.prev_gain_r = [1.0] * length
        self.count_l = 0
        self.count_r = 0
        self.sum_gain_l = float(length)
        self.sum_gain_r = float(length)

    def set_detection_type(self, detection_type):
        self.detector.set_detection_type(detection_type)
        self.set_attack_time(self.attack_time)
        self.set_release_time(self.release_time)

    def process_channel(self, sample, channel, prev_gain, count, sum_gain):
        env = self.detector.process_sample(sample, channel)
        gain = 1.0 if env < self.threshold else (self.threshold_inv * env) ** self.precomputed_exp

        sum_gain -= prev_gain[count]
        prev_gain[count] = gain
        count += 1
        if count == self.prev_gain_size:
            count = 0
        sum_gain += gain

        return sample * (sum_gain * self.inv_prev_gain_size), prev_gain, count, sum_gain

    def process(self, buffer):
        # buffer shape: (num_frames, 2) for stereo
        output = np.zeros_like(buffer)
        for i, (l, r) in enumerate(buffer):
            l_out, self.prev_gain_l, self.count_l, self.sum_gain_l = self.process_channel(
                l, 0, self.prev_gain_l, self.count_l, self.sum_gain_l
            )
            r_out, self.prev_gain_r, self.count_r, self.sum_gain_r = self.process_channel(
                r, 1, self.prev_gain_r, self.count_r, self.sum_gain_r
            )
            output[i, 0] = l_out
            output[i, 1] = r_out
        return output