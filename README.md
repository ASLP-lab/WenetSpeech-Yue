📢：**Good news! 8000 hours of multi-label Wu dialect data are also available at [⭐WenetSpeech-Wu⭐](https://github.com/ASLP-lab/WenetSpeech-Wu-Repo).**
📢：**Good news! 10,000 hours of multi-label Chuan-Yu speech data are also available at [⭐WenetSpeech-Chuan⭐](https://github.com/ASLP-lab/WenetSpeech-Chuan).**

# WenetSpeech-Yue: A Large-scale Cantonese Speech Corpus with Multi-dimensional Annotation

<p align="center">
  Longhao Li<sup>1</sup>*, Zhao Guo<sup>1</sup>*, Hongjie Chen<sup>2</sup>, 
  Yuhang Dai<sup>1</sup>, Ziyu Zhang<sup>1</sup>, Hongfei Xue<sup>1</sup>, 
  Tianlun Zuo<sup>1</sup>, Chengyou Wang<sup>1</sup>, Shuiyuan Wang<sup>1</sup>, 
  Xin Xu<sup>3</sup>, Hui Bu<sup>3</sup>, Jie Li<sup>2</sup>, Jian Kang<sup>2</sup>, 
  Binbin Zhang<sup>4</sup>, Ruibin Yuan<sup>5</sup>, Ziya Zhou<sup>5</sup>, 
  Wei Xue<sup>5</sup>, Lei Xie<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University <br>
  <sup>2</sup> Institute of Artificial Intelligence (TeleAI), China Telecom <br>
  <sup>3</sup> Beijing AISHELL Technology Co., Ltd. <br>
  <sup>4</sup> WeNet Open Source Community <br>
  <sup>5</sup> Hong Kong University of Science and Technology
</p>


<p align="center">
📑 <a href="https://arxiv.org/abs/2509.03959">Paper</a> &nbsp&nbsp | &nbsp&nbsp 
🐙 <a href="https://github.com/ASLP-lab/WenetSpeech-Yue">GitHub</a> &nbsp&nbsp | &nbsp&nbsp 
🤗 <a href="https://huggingface.co/collections/ASLP-lab/wenetspeech-yue-68b690d287cde88389e5dba1">HuggingFace</a>
<br>
🖥️ <a href="https://huggingface.co/spaces/ASLP-lab/WenetSpeech-Yue">HuggingFace Space</a> &nbsp&nbsp | &nbsp&nbsp 
🎤 <a href="https://aslp-lab.github.io/WenetSpeech-Yue/">Demo Page</a> &nbsp&nbsp | &nbsp&nbsp 
💬 <a href="https://github.com/ASLP-lab/WenetSpeech-Yue?tab=readme-ov-file#contact">Contact Us</a>
</p>


This is the official repository 👑 for the WenetSpeech-Yue dataset and the source code for WenetSpeech-Pipe speech data preprocessing pipeline.

<div align="center"><img width="800px" src="https://github.com/ASLP-lab/WenetSpeech-Yue/blob/main/figs/wenetspeech_yue.svg" /></div>

## 📢 News and Updates
- **2026/01/07**: 🛠️ We fixed several issues in the metadata. Please use the latest metadata for training.
- **2025/11/15** 🚀 We released **Llasa-1B-Yue-Updated**！ You can download the model weights from [WSYue-TTS](https://huggingface.co/ASLP-lab/WSYue-TTS)
- **2025/09/08**: 🎉 The WenetSpeechYue dataset, featuring over 21,800 hours of Cantonese speech, is now available!

## Download
* The WenetSpeech-Yue dataset is available at [WenetSpeech-Yue](https://huggingface.co/datasets/ASLP-lab/WenetSpeech-Yue).
* The WSYue-eval benchmark is available at [WSYue-ASR-eval](https://huggingface.co/datasets/ASLP-lab/WSYue-ASR-eval) for ASR and [WSYue-TTS-eval](https://huggingface.co/datasets/ASLP-lab/WSYue-ASR-eval) for TTS.
* The ASR models are available at [WSYue-ASR](https://huggingface.co/ASLP-lab/WSYue-ASR).
* The TTS models are available at [WSYue-TTS](https://huggingface.co/ASLP-lab/WSYue-TTS).



## Dataset
### WenetSpeech-Yue Overview
* Contains 21,800 hours of large-scale Cantonese speech corpus with rich annotations, the largest open-source resource for Cantonese speech research.
* Stores metadata in a single JSON file, including audio path, duration, text confidence, speaker identity, SNR, DNSMOS, age, gender, and character-level timestamps. Additional metadata tags may be added in the future.
* Covers ten domains: Storytelling, Entertainment, Drama, Culture, Vlog, Commentary, Education, Podcast, News, and Others.
<div align="center"><img width="800px" src="https://github.com/ASLP-lab/WenetSpeech-Yue/blob/main/figs/data_distribution.png" /></div>

## Benchmark
To address the unique linguistic characteristics of Cantonese, we propose WSYue-eval, a comprehensive benchmark encompassing both Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) tasks.

### ASR Benchmark
We introduce WSYue-ASR-eval, a test set developed for Automatic Speech Recognition (ASR) as a key task in speech understanding. It features **multi-round manual annotations** including text transcripts, emotion, age, and gender labels. The set is divided into Short and Long subsets by audio duration to enable comprehensive evaluation across speech lengths. WSYue-ASR-eval also covers diverse real-world Cantonese scenarios, including code-switching and multi-domain conditions.

| Set   | Duration | Speakers | Hours |
|-------|----------|----------|-------|
| Short | 0–10s    | 2861     | 9.46  |
| Long  | 10–30s   | 838      | 1.97  |

### TTS Benchmark
We introduce WSYue-TTS-eval, a zero-shot Cantonese TTS benchmark with two subsets:
- Base: Contains 1,000 samples from Common Voice for evaluating real-world performance.
- Coverage: Combines manually curated and LLM-generated texts spanning multiple domains (e.g., daily life, news, entertainment, poetry) and incorporates diverse linguistic phenomena including polyphonic characters, tone sandhi, code-switching, proper nouns, and numerals.


## ASR Leaderboard
<table border="0" cellspacing="0" cellpadding="6" style="border-collapse:collapse;">
  <tr>
    <th align="left" rowspan="2">Model</th>
    <th align="center" rowspan="2">#Params (M)</th>
    <th align="center" colspan="2">In-House</th>
    <th align="center" colspan="5">Open-Source</th>
    <th align="center" colspan="2">WSYue-eval</th>
  </tr>
  <tr>
    <th align="center">Dialogue</th>
    <th align="center">Reading</th>
    <th align="center">yue</th>
    <th align="center">HK</th>
    <th align="center">MDCC</th>
    <th align="center">Daily_Use</th>
    <th align="center">Commands</th>
    <th align="center">Short</th>
    <th align="center">Long</th>
  </tr>

  <tr><td align="left" colspan="11"><b>w/o LLM</b></td></tr>
  <tr>
    <td align="left"><a href="https://huggingface.co/ASLP-lab/WSYue-ASR/tree/main/u2pp_conformer_yue"><b>Conformer-Yue</b></a>⭐</td><td align="center">130</td><td align="center"><b>16.57</b></td><td align="center">7.82</td><td align="center">7.72</td><td align="center">11.42</td><td align="center">5.73</td><td align="center">5.73</td><td align="center">8.97</td><td align="center"><ins>5.05</ins></td><td align="center">8.89</td>
  </tr>
  <tr>
    <td align="left">Paraformer</td><td align="center">220</td><td align="center">83.22</td><td align="center">51.97</td><td align="center">70.16</td><td align="center">68.49</td><td align="center">47.67</td><td align="center">79.31</td><td align="center">69.32</td><td align="center">73.64</td><td align="center">89.00</td>
  </tr>
  <tr>
    <td align="left">SenseVoice-small</td><td align="center">234</td><td align="center">21.08</td><td align="center"><ins>6.52</ins></td><td align="center">8.05</td><td align="center"><b>7.34</b></td><td align="center">6.34</td><td align="center">5.74</td><td align="center"><ins>6.65</ins></td><td align="center">6.69</td><td align="center">9.95</td>
  <tr>
    <td align="left"><a href="https://huggingface.co/ASLP-lab/WSYue-ASR/tree/main/sensevoice_small_yue"><b>SenseVoice-s-Yue</b></a>⭐</td><td align="center">234</td><td align="center">19.19</td><td align="center">6.71</td><td align="center">6.87</td><td align="center">8.68</td><td align="center"><ins>5.43</ins></td><td align="center">5.24</td><td align="center">6.93</td><td align="center">5.23</td><td align="center">8.63</td>
  </tr>
  </tr>
  <tr>
    <td align="left">Dolphin-small</td><td align="center">372</td><td align="center">59.20</td><td align="center">7.38</td><td align="center">39.69</td><td align="center">51.29</td><td align="center">26.39</td><td align="center">7.21</td><td align="center">9.68</td><td align="center">32.32</td><td align="center">58.20</td>
  </tr>
  <tr>
    <td align="left">TeleASR</td><td align="center">700</td><td align="center">37.18</td><td align="center">7.27</td><td align="center">7.02</td><td align="center"><ins>7.88</ins></td><td align="center">6.25</td><td align="center">8.02</td><td align="center"><b>5.98</b></td><td align="center">6.23</td><td align="center">11.33</td>
  </tr>
  <tr>
    <td align="left">Whisper-medium</td><td align="center">769</td><td align="center">75.50</td><td align="center">68.69</td><td align="center">59.44</td><td align="center">62.50</td><td align="center">62.31</td><td align="center">64.41</td><td align="center">80.41</td><td align="center">80.82</td><td align="center">50.96</td>
  </tr>
  <tr>
    <td align="left"><a href="https://huggingface.co/ASLP-lab/WSYue-ASR/tree/main/whisper_medium_yue"><b>Whisper-m-Yue</b></a>⭐</td><td align="center">769</td><td align="center">18.69</td><td align="center">6.86</td><td align="center"><ins>6.86</ins></td><td align="center">11.03</td><td align="center">5.49</td><td align="center"><ins>4.70</ins></td><td align="center">8.51</td><td align="center"><ins>5.05</ins></td><td align="center"><ins>8.05</ins></td>
  </tr>

  <tr>
    <td align="left">FireRedASR-AED-L</td><td align="center">1100</td><td align="center">73.70</td><td align="center">18.72</td><td align="center">43.93</td><td align="center">43.33</td><td align="center">34.53</td><td align="center">48.05</td><td align="center">49.99</td><td align="center">55.37</td><td align="center">50.26</td>
  </tr>
  <tr>
    <td align="left">Whisper-large-v3</td><td align="center">1550</td><td align="center">45.09</td><td align="center">15.46</td><td align="center">12.85</td><td align="center">16.36</td><td align="center">14.63</td><td align="center">17.84</td><td align="center">20.70</td><td align="center">12.95</td><td align="center">26.86</td>
  </tr>

  <tr><td align="left" colspan="11"><b>w/ LLM</b></td></tr>

  <tr>
    <td align="left">Qwen2.5-Omni-3B</td><td align="center">3000</td><td align="center">72.01</td><td align="center">7.49</td><td align="center">12.59</td><td align="center">11.75</td><td align="center">38.91</td><td align="center">10.59</td><td align="center">25.78</td><td align="center">67.95</td><td align="center">88.46</td>
  </tr>
  <tr>
    <td align="left">Kimi-Audio</td><td align="center">7000</td><td align="center">68.65</td><td align="center">24.34</td><td align="center">40.90</td><td align="center">38.72</td><td align="center">30.72</td><td align="center">44.29</td><td align="center">45.54</td><td align="center">50.86</td><td align="center">33.49</td>
  </tr>
  <tr>
    <td align="left">FireRedASR-LLM-L</td><td align="center">8300</td><td align="center">73.70</td><td align="center">18.72</td><td align="center">43.93</td><td align="center">43.33</td><td align="center">34.53</td><td align="center">48.05</td><td align="center">49.99</td><td align="center">49.87</td><td align="center">45.92</td>
  </tr>
  <tr>
    <td align="left"><b>Conformer-LLM-Yue⭐</b></td><td align="center">4200</td><td align="center"><ins>17.22</ins></td><td align="center"><b>6.21</b></td><td align="center"><b>6.23</b></td><td align="center">9.52</td><td align="center"><b>4.35</b></td><td align="center"><b>4.57</b></td><td align="center">6.98</td><td align="center"><b>4.73</b></td><td align="center"><b>7.91</b></td>
  </tr>
</table>

## ASR Inference
### U2pp_Conformer_Yue
```
dir=u2pp_conformer_yue
decode_checkpoint=$dir/u2pp_conformer_yue.pt
test_set=path/to/test_set
test_result_dir=path/to/test_result_dir

python wenet/bin/recognize.py \
  --gpu 0 \
  --modes attention_rescoring \
  --config $dir/train.yaml \
  --test_data $test_set/data.list \
  --checkpoint $decode_checkpoint \
  --beam_size 10 \
  --batch_size 32 \
  --ctc_weight 0.5 \
  --result_dir $test_result_dir \
  --decoding_chunk_size -1
```
### Whisper_Medium_Yue
```
dir=whisper_medium_yue
decode_checkpoint=$dir/whisper_medium_yue.pt
test_set=path/to/test_set
test_result_dir=path/to/test_result_dir

python wenet/bin/recognize.py \
  --gpu 0 \
  --modes attention \
  --config $dir/train.yaml \
  --test_data $test_set/data.list \
  --checkpoint $decode_checkpoint \
  --beam_size 10 \
  --batch_size 32 \
  --blank_penalty 0.0 \
  --ctc_weight 0.0 \
  --reverse_weight 0.0 \
  --result_dir $test_result_dir \
  --decoding_chunk_size -1
```
### SenseVoice_Small_Yue
```
from funasr import AutoModel

model_dir = "sensevoice_small_yue"

model = AutoModel(
        model=model_path,
        device="cuda:0",
    )
res = model.generate(
    wav_path,
    cache={},
    language="yue",
    use_itn=True,
    batch_size=64,
)
```

## TTS Inference
### Install

**Clone and install**

- Clone the repo
``` sh
git clone https://github.com/ASLP-lab/WenetSpeech-Yue.git
cd CosyVoice2-Yue
```

- Create Conda env:

``` sh
conda create -n cosyvoice python=3.10
conda activate cosyvoice
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

```

### Model download

``` python
from huggingface_hub import snapshot_download
snapshot_download('ASLP-lab/WSYue-TTS', local_dir='pretrained_models')
```

### Usage


``` python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import opencc

# s2t
converter = opencc.OpenCC('s2t.json')

cosyvoice_base = CosyVoice2(
    'ASLP-lab/Cosyvoice2-Yue',
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)

cosyvoice_zjg = CosyVoice2(
    'ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai',
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)

prompt_speech_16k = load_wav('asset/sg_017_090.wav', 16000)

text = '收到朋友从远方寄嚟嘅生日礼物，嗰份意外嘅惊喜同埋深深嘅祝福令我心入面充满咗甜蜜嘅快乐，笑容好似花咁绽放。'
text = converter.convert(text)

for i, j in enumerate(cosyvoice_base.inference_instruct2(text, '用粤语说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('base_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice_zjg.inference_instruct2(text, '用粤语说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('zjg_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

## WenetSpeech-Pipe

WenetSpeech-Pipe Overview:
<div align="center"><img width="800px" src="https://github.com/ASLP-lab/WenetSpeech-Yue/blob/main/figs/wenetspeech_pipe.svg" /></div>

### Audio Collection
WenetSpeech-Pipe collects large-scale, in-the-wild speech recordings across diverse domains such as storytelling, drama, commentary, vlogs, food, entertainment, news, and education. These long recordings are segmented into short clips with VAD, yielding utterance-level data for transcription and quality evaluation.
<div align="center"><img width="600px" src="https://github.com/ASLP-lab/WenetSpeech-Yue/blob/main/figs/domain_distribution.svg" /></div>

### Speaker Attribute Annotation
To enrich the dataset with speaker-level metadata for multi-speaker modeling and style-aware synthesis, WenetSpeech-Pipe includes a Speaker Attributes Annotation stage. Using [pyannote](https://github.com/pyannote/pyannote-audio) toolkit for speaker diarization and [Vox-Profile](https://github.com/tiantiaf0627/vox-profile-release) for age and gender estimation, each utterance-level segment is annotated with speaker identity, age, and gender, enabling supervised and style-controllable speech modeling.

### Speech Quality Annotation
To support high-fidelity tasks such as TTS and voice conversion, WenetSpeech-Pipe integrates a comprehensive quality assessment stage. Each segment is evaluated by (i) [Brouhaha](https://github.com/marianne-m/brouhaha-vad) for signal-to-noise ratio (SNR), (ii) [DNSMOS](https://github.com/microsoft/DNS-Challenge) for perceptual quality (MOS), and (iii) bandwidth detection for spectral coverage. These complementary measures yield structured annotations with quantitative scores and spectral references.

### Automatic Speech Recognition
We select three models with the best performance on Cantonese to perform multi-system labeling: SenseVoice, TeleASR, and Whisper. For each audio file, we obtain the corresponding multi-system transcriptions.

### Text Postprocessing
Each ASR transcription system produces outputs in different formats. To standardize these formats, we introduce a text post-processing module, which includes punctuation removal, traditional-to-simplified Chinese conversion, and text normalization. The detailed code can be found in `text_postprocessing.py`.
<div align="center"><img width="300px" src="https://github.com/ASLP-lab/WenetSpeech-Yue/blob/main/figs/text_processing.svg" /></div>

### Recognizer Output Voting
Despite text postprocessing, ASR outputs still vary in lexical choice, segmentation, and phonetic representation. To obtain unified and accurate reference transcriptions, we adopt and extend the ROVER framework for Cantonese. Normalized transcriptions are first aligned using dynamic programming, after which a filtering module removes outlier outputs based on edit distance. Voting then selects the most frequent word at each aligned position, and the average voting frequency is recorded as an utterance-level confidence score. In parallel, we extend the voting mechanism to Cantonese pinyin by introducing a pronunciation-level confidence measure, further reinforcing phoneme consistency.

To further enhance transcription accuracy, we leverage [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) for minimal, context-aware refinements of the consensus output. The LLM references all original ASR hypotheses and applies only essential corrections—such as grammar, lexical choice, or named entities—while preserving the integrity of the spoken content.

<div align="center"><img width="500px" src="https://github.com/ASLP-lab/WenetSpeech-Yue/blob/main/figs/llm_corrector.svg" /></div>

## Contributors

| <img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/nwpu.png" width="200px"> | <img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/aishelltech.png" width="200px"> | <img src="https://raw.githubusercontent.com/ASLP-lab/WenetSpeech-Yue/main/figs/teleai.png" width="200px"> | <img src="https://raw.githubusercontent.com/ASLP-lab/WenetSpeech-Yue/main/figs/wenet.png" width="200px"> | <img src="https://raw.githubusercontent.com/ASLP-lab/WenetSpeech-Yue/main/figs/hkust.jpg" width="200px"> |
| ---- | ---- | ---- | ---- | ---- |



## Citation
Please cite our paper if you find this work useful:
```
@misc{li2025wenetspeechyuelargescalecantonesespeech,
      title={WenetSpeech-Yue: A Large-scale Cantonese Speech Corpus with Multi-dimensional Annotation}, 
      author={Longhao Li and Zhao Guo and Hongjie Chen and Yuhang Dai and Ziyu Zhang and Hongfei Xue and Tianlun Zuo and Chengyou Wang and Shuiyuan Wang and Jie Li and Xin Xu and Hui Bu and Binbin Zhang and Ruibin Yuan and Ziya Zhou and Wei Xue and Lei Xie},
      year={2025},
      eprint={2509.03959},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2509.03959}, 
}
```

## Contact
If you are interested in leaving a message to our research team, feel free to email lhli@mail.nwpu.edu.cn or gzhao@mail.nwpu.edu.cn.
<p align="center">
<img src="./figs/wechat.jpg" width="300" alt="WeChat Group QR Code"/>
<br>
<em>Scan to join our WeChat discussion group</em>
<br>
</p>
<p align="center">
    <img src="./figs/npu@aslp.jpeg" width="500"/>
</p>
