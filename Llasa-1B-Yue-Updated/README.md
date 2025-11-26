本项目包含 Llasa-1B-Yue 的推理代码与配置文件，但 不包含 BiCodec 模型权重（model.safetensors）。
由于 GitHub 的单文件 100MB 限制，需要用户手动下载对应权重文件。

请从以下链接下载对应的 BiCodec 模型权重（model.safetensors）：[bicodec-sparkTTS](https://huggingface.co/SparkAudio/Spark-TTS-0.5B/tree/main)

下载后将权重文件手动放到：
```
Llasa-1B-Yue-Updated/bicodec_ckpt/ckpt/model.safetensors
```

This project includes the inference code and configuration files for Llasa-1B-Yue, but does not include the BiCodec model weights (model.safetensors).
Due to GitHub’s 100MB file size limit, users must download the required weights manually.

Please download the BiCodec model weights (model.safetensors) from the following link:
[bicodec-sparkTTS](https://huggingface.co/SparkAudio/Spark-TTS-0.5B/tree/main)

After downloading, place the weight file at:
```
Llasa-1B-Yue-Updated/bicodec_ckpt/ckpt/model.safetensors
```