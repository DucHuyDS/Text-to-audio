import sys

sys.path.append("/content/gdrive/MyDrive/Colab Notebooks/data/zalo_ai/")

import os
import pandas as pd
import yaml
import audioldm_train.utilities.audio as Audio
from audioldm_train.utilities.tools import load_json
from audioldm_train.dataset_plugin import *
from librosa.filters import mel as librosa_mel_fn

import random
from torch.utils.data import Dataset
import torch.nn.functional
import torch
import numpy as np
import torchaudio
import json


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class AudioDataset(Dataset):
    def __init__(
        self,
        config=None,
        split="train",
        waveform_only=False,
        add_ons=[],
        dataset_json=None,
        val = False
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.config = config
        self.split = split
        self.val = val
        self.pad_wav_start_sample = 0  # If none, random choose
        self.trim_wav = False
        self.waveform_only = waveform_only
        self.add_ons = [eval(x) for x in add_ons]
        print("Add-ons:", self.add_ons)

        self.build_setting_parameters()
        self.dataset_name = self.config["data"][self.split]
        if dataset_json is not None:
            self.data = dataset_json["data"]
        else:
          self.build_dataset()
        self.id2label, self.index_dict, self.num2label = {}, {}, {}


        self.build_dsp()
        self.label_num = len(self.index_dict)
        print("Dataset initialize finished")
        # asd

    def __getitem__(self, index):
        (
            fname,
            waveform,
            stft,
            log_mel_spec,
            label_vector,  # the one-hot representation of the audio class
            # the metadata of the sampled audio file and the mixup audio file (if exist)
            (datum, mix_datum),
            random_start,
        ) = self.feature_extraction(index)

        # text = self.get_sample_text_caption(datum, mix_datum, label_vector)

        text = self.data[index]['caption']

        
        data = {
            "text": text,  # list
            "fname": fname,  # list
            # tensor, [batchsize, class_num]
            "label_vector": "" if (label_vector is None) else label_vector.float(),
            # tensor, [batchsize, 1, samples_num]
            "waveform": "" if (waveform is None) else waveform.float(),
            # tensor, [batchsize, t-steps, f-bins]
            "stft": "" if (stft is None) else stft.float(),
            # tensor, [batchsize, t-steps, mel-bins]
            "log_mel_spec": "" if (log_mel_spec is None) else log_mel_spec.float(),
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "random_start_sample_in_original_audio_file": random_start,
        }

        for add_on in self.add_ons:
            data.update(add_on(self.config, data, self.data[index]))

        if data["text"] is None:
            print("Warning: The model return None on key text", fname)
            data["text"] = ""

        return data

    def __len__(self):
        return len(self.data)

    def feature_extraction(self, index):
        while True:
            try:
                # label_indices = np.zeros(self.label_num, dtype=np.float32)
                datum = self.data[index]
                (
                    log_mel_spec,
                    stft,
                    waveform,
                    random_start,
                ) = self.read_audio_file(datum["wav"])
                mix_datum = None

                label_indices = torch.FloatTensor(0)
                break
            except Exception as e:
                index = (index + 1) % len(self.data)
                print(
                    "Error encounter during audio feature extraction: ", e, datum["wav"]
                )
                raise  Exception("bi loi o feature_extraction")
                continue

        fname = datum["wav"]
        waveform = torch.FloatTensor(waveform)

        return (
            fname,
            waveform,
            stft,
            log_mel_spec,
            label_indices,
            (datum, mix_datum),
            random_start,
        )


    def build_setting_parameters(self):
        # Read from the json config
        self.melbins = self.config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = self.config["preprocessing"]["stft"]["hop_length"]
        self.duration = self.config["preprocessing"]["audio"]["duration"]
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)

        self.mixup = self.config["augmentation"]["mixup"]


        if "test" in self.split or self.val == True: 
          self.mixup = 0.0

    def build_dataset(self):
        self.data = []

        with open(f"data/dataset/{self.split}.json", 'r') as file:
            labels = json.load(file)
        data_audio_dir = "data/dataset/audio"
        dir_map = os.listdir(data_audio_dir)
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if "(1)" not in name:
                if ext == ".mp3":
                    self.data.append({
                        'wav':os.path.join(data_audio_dir, d),
                        "caption" : labels[d]
                    })

                else:
                    raise ValueError(f"No label file for {name}")

        if self.val == True:
            self.data = self.data[int(len(self.data)*0.8):]
        print("Data size: {}".format(len(self.data)))

    def build_dsp(self):
        self.mel_basis = {}
        self.hann_window = {}

        self.filter_length = self.config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = self.config["preprocessing"]["stft"]["hop_length"]
        self.win_length = self.config["preprocessing"]["stft"]["win_length"]
        self.n_mel = self.config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.mel_fmin = self.config["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = self.config["preprocessing"]["mel"]["mel_fmax"]

        self.STFT = Audio.stft.TacotronSTFT(
            self.config["preprocessing"]["stft"]["filter_length"],
            self.config["preprocessing"]["stft"]["hop_length"],
            self.config["preprocessing"]["stft"]["win_length"],
            self.config["preprocessing"]["mel"]["n_mel_channels"],
            self.config["preprocessing"]["audio"]["sampling_rate"],
            self.config["preprocessing"]["mel"]["mel_fmin"],
            self.config["preprocessing"]["mel"]["mel_fmax"],
        )

    def resample(self, waveform, sr):
        waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        return waveform

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

    def random_segment_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        for i in range(10):
            random_start = int(self.random_uniform(0, waveform_length - target_length))
            if torch.max(
                torch.abs(waveform[:, random_start : random_start + target_length])
                > 1e-4
            ):
                break

        return waveform[:, random_start : random_start + target_length], random_start

    def pad_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == target_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        if self.pad_wav_start_sample is None:
            rand_start = int(self.random_uniform(0, target_length - waveform_length))
        else:
            rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length
            while start - chunk_size > 0:
                if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                    start -= chunk_size
                else:
                    break
            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    def read_wav_file(self, filename):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        waveform, sr = torchaudio.load(filename)

        waveform, random_start = self.random_segment_wav(
            waveform, target_length=int(sr * self.duration)
        )

        waveform = self.resample(waveform, sr)
        # random_start = int(random_start * (self.sampling_rate / sr))

        waveform = waveform.numpy()[0, ...]

        waveform = self.normalize_wav(waveform)

        if self.trim_wav:
            waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]
        waveform = self.pad_wav(
            waveform, target_length=int(self.sampling_rate * self.duration)
        )
        return waveform, random_start

    def read_audio_file(self, filename, filename2=None):
        if os.path.exists(filename):
            waveform, random_start = self.read_wav_file(filename)
        else:
            print("tam biet nhe nguoi anh yeu")
            print(
                'Non-fatal Warning [dataset.py]: The wav path "',
                filename,
                '" is not find in the metadata. Use empty waveform instead. This is normal in the inference process.',
            )
            target_length = int(self.sampling_rate * self.duration)
            waveform = torch.zeros((1, target_length))
            random_start = 0

        # log_mel_spec, stft = self.wav_feature_extraction_torchaudio(waveform) # this line is faster, but this implementation is not aligned with HiFi-GAN
        if not self.waveform_only:
            log_mel_spec, stft = self.wav_feature_extraction(waveform)
        else:
            # Load waveform data only
            # Use zero array to keep the format unified
            log_mel_spec, stft = None, None

        return log_mel_spec, stft, waveform, random_start



    def mel_spectrogram_train(self, y):
        if torch.min(y) < -1.0:
            print("train min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("train max value is ", torch.max(y))

        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                sr = self.sampling_rate,
                n_fft = self.filter_length,
                n_mels = self.n_mel,
                fmin = self.mel_fmin,
                fmax = self.mel_fmax,
            )
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(
                y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )

        y = y.squeeze(1)

        stft_spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(y.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        stft_spec = torch.abs(stft_spec)

        mel = spectral_normalize_torch(
            torch.matmul(
                self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec
            )
        )

        return mel[0], stft_spec[0]

    # This one is significantly slower than "wav_feature_extraction_torchaudio" if num_worker > 1
    def wav_feature_extraction(self, waveform):
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        # log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)[0]
        log_mel_spec, stft = self.mel_spectrogram_train(waveform.unsqueeze(0))

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        return log_mel_spec, stft


    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec


    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def frequency_masking(self, log_mel_spec, freqm):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq - mask_len))
        log_mel_spec[:, mask_start : mask_start + mask_len, :] *= 0.0
        return log_mel_spec

    def time_masking(self, log_mel_spec, timem):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps - mask_len))
        log_mel_spec[:, :, mask_start : mask_start + mask_len] *= 0.0
        return log_mel_spec


if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader

    seed_everything(0)

    def write_json(my_dict, fname):
        # print("Save json file at "+fname)
        json_str = json.dumps(my_dict)
        with open(fname, "w") as json_file:
            json_file.write(json_str)

    def load_json(fname):
        with open(fname, "r") as f:
            data = json.load(f)
            return data

    config = yaml.load(
        open(
            "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/config/vae_48k_256/ds_8_kl_1.0_ch_16.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )

    add_ons = config["data"]["dataloader_add_ons"]

    # load_json(data)
    dataset = AudioDataset(
        config=config, split="train", waveform_only=False, add_ons=add_ons
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    for cnt, each in tqdm(enumerate(loader)):
        # print(each["waveform"].size(), each["log_mel_spec"].size())
        # print(each['freq_energy_percentile'])
        import ipdb

        ipdb.set_trace()
        # pass
