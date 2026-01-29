# dNoize

Meaningful Speech Enhancement (Noise and Reverb Suppression) | Native-Friendly to torch_directml on Windows 10/11

# VoiceBank-DEMAND Dataset Setup

This guide explains how to download and prepare the VoiceBank-DEMAND dataset for audio denoising training.

## Dataset Information

**Source:** University of Edinburgh DataShare  
**URL:** https://datashare.ed.ac.uk/handle/10283/2791

The VoiceBank-DEMAND dataset contains clean and noisy speech recordings for training speech enhancement models.

## Downloads

Download the following files from the dataset page:

### Training Data (28 speakers)
- `clean_trainset_28spk_wav.zip` (2.315 GB)
- `noisy_trainset_28spk_wav.zip` (2.635 GB)

### Test Data
- `clean_testset_wav.zip` (147.1 MB)
- `noisy_testset_wav.zip` (162.6 MB)

### Optional: Larger Training Set (56 speakers)
- `clean_trainset_56spk_wav.zip` (4.442 GB)
- `noisy_trainset_56spk_wav.zip` (5.240 GB)

## Directory Structure

After downloading, organize the files in the following structure:

```
project_root/
├── data/
│   ├── train/
│   │   ├── clean_trainset_28spk_wav/
│   │   │   ├── p226_001.wav
│   │   │   ├── p226_002.wav
│   │   │   └── ...
│   │   └── noisy_trainset_28spk_wav/
│   │       ├── p226_001.wav
│   │       ├── p226_002.wav
│   │       └── ...
│   └── test/
│       ├── clean_testset_wav/
│       │   ├── p232_001.wav
│       │   ├── p232_002.wav
│       │   └── ...
│       └── noisy_testset_wav/
│           ├── p232_001.wav
│           ├── p232_002.wav
│           └── ...
└── ...
```

## Setup Instructions

### Step 1: Create Directory Structure

```bash
mkdir -p data/train
mkdir -p data/test
```

### Step 2: Extract Training Data

```bash
# Extract clean training audio
unzip clean_trainset_28spk_wav.zip -d data/train/

# Extract noisy training audio
unzip noisy_trainset_28spk_wav.zip -d data/train/
```

### Step 3: Extract Test Data

```bash
# Extract clean test audio
unzip clean_testset_wav.zip -d data/test/

# Extract noisy test audio
unzip noisy_testset_wav.zip -d data/test/
```

## Dataset Statistics

### Training Set (28 speakers)
- Clean files: 11,572 audio files
- Noisy files: 11,572 audio files
- Total speakers: 28
- Sample rate: 48 kHz (will be resampled to 16 kHz during training)

### Test Set
- Clean files: 824 audio files
- Noisy files: 824 audio files
- Total speakers: 2
- Sample rate: 48 kHz (will be resampled to 16 kHz during inference)

## File Naming Convention

Files are paired by name:
- Clean: `data/train/clean_trainset_28spk_wav/p226_001.wav`
- Noisy: `data/train/noisy_trainset_28spk_wav/p226_001.wav`

The naming format is `p{speaker_id}_{utterance_id}.wav`


## Troubleshooting

### Issue: Sample rate errors
The dataset uses 48 kHz audio, but the model expects 16 kHz. This is handled automatically during training:
```python
# In load_audio function (already implemented):
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    wave = resampler(wave)
```

### Issue: Extraction errors on Windows
Use 7-Zip or WinRAR if the built-in Windows extractor fails:
```
Right-click zip file -> 7-Zip -> Extract to "data/train/"
```

## License and Citation

If you use this dataset, please cite:

```
Valentini-Botinhao, C. (2017). Noisy speech database for training speech enhancement 
algorithms and TTS models [Data set]. University of Edinburgh. School of Informatics. 
Centre for Speech Technology Research (CSTR). https://doi.org/10.7488/ds/2117
```

## Additional Requirements

Before installing the python packages using requirements.txt, you need to download and install Windows 11 SDK (latest version) using the [Visual Studio Installer Tool](https://visualstudio.microsoft.com/vs/community/). This is required for building pesq package. If you are not going to use pesq, you can skip it but do not forget to remove it from the train code.