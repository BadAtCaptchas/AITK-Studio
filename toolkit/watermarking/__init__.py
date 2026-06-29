from .codecs import BUILTIN_AUTHENLORA_CODECS, get_builtin_codec_options, resolve_codec_path

__all__ = [
    "AuthenLoRAController",
    "MapperNet",
    "SecretDecoder",
    "SecretEncoder",
    "bit_accuracy",
    "BUILTIN_AUTHENLORA_CODECS",
    "get_builtin_codec_options",
    "resolve_codec_path",
]


def __getattr__(name):
    if name in {"AuthenLoRAController", "MapperNet", "SecretDecoder", "SecretEncoder", "bit_accuracy"}:
        from .authenlora import AuthenLoRAController, MapperNet, SecretDecoder, SecretEncoder, bit_accuracy

        values = {
            "AuthenLoRAController": AuthenLoRAController,
            "MapperNet": MapperNet,
            "SecretDecoder": SecretDecoder,
            "SecretEncoder": SecretEncoder,
            "bit_accuracy": bit_accuracy,
        }
        return values[name]
    raise AttributeError(name)
