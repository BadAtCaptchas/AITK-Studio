# AuthenLoRA Built-In Codecs

These bundled codec checkpoints are copied from the official AuthenLoRA Hugging Face repository:

- `shifangming0823/AuthenLoRA_model`
- `codec_48bits.pth`
- `Extra/codec_80bits.pth`
- `Extra/codec_100bits.pth`

They are resolved by config IDs:

- `builtin:authenlora_48bits`
- `builtin:authenlora_80bits`
- `builtin:authenlora_100bits`

Training still works with an explicit local `watermark.codec_path`; built-in IDs are just packaged local defaults.
