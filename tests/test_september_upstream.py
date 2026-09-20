"""Weight-free coverage of the September model/inference integration."""
import json
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from toolkit.config_modules import GenerateImageConfig, DatasetConfig
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.dataloader_mixins import LatentCachingMixin
from toolkit.dto import DTO
from toolkit.inference_lora import LoRAStack
from toolkit.util.streamed_safetensors import save_file_streamed
from toolkit.util.comfy_quant_import import import_comfy_quantized_layers
from toolkit.util.convrot_quant import rotate
from toolkit.audio.album_artwork import load_waveform, create_artwork
from extensions_built_in.inference_engine.protocol import FrameReader
from extensions_built_in.inference_engine.engine import Engine, EngineBusy, GenerationJob, END, _list_outputs, _kind_for_ext
from extensions_built_in.inference_engine.validation import validate_generation


class SeptemberUpstreamTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='aitk-upstream-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_streamed_checkpoint_roundtrip_all_shapes_and_types(self):
        tensors = {
            'scalar': torch.tensor(1.25), 'empty': torch.empty(0, 2),
            'noncontiguous': torch.arange(12).reshape(3, 4).T,
            'bf16': torch.randn(2, 3).bfloat16(), 'bool': torch.tensor([True, False]),
            'int8': torch.tensor([-128, 127], dtype=torch.int8),
        }
        filename = self.root / 'checkpoint.safetensors'
        with patch('toolkit.util.streamed_safetensors._CHUNK_BYTES', 3):
            save_file_streamed(tensors, filename, {'source': 'test'})
        actual = load_file(filename)
        for name, tensor in tensors.items():
            self.assertEqual(actual[name].dtype, tensor.dtype)
            self.assertTrue(torch.equal(actual[name], tensor))
        with safe_open(filename, framework='pt') as handle:
            self.assertEqual(handle.metadata(), {'source': 'test'})

    def test_failed_checkpoint_publish_preserves_previous_file(self):
        filename = self.root / 'checkpoint.safetensors'
        filename.write_bytes(b'original')
        with patch('toolkit.util.streamed_safetensors.os.replace', side_effect=OSError('full disk')):
            with self.assertRaises(OSError): save_file_streamed({'x': torch.ones(4)}, filename)
        self.assertEqual(filename.read_bytes(), b'original')
        self.assertEqual([p.name for p in self.root.iterdir()], [filename.name])

    def test_rotated_embedding_import_and_preflight(self):
        root = torch.nn.Module(); root.embed = torch.nn.Embedding(3, 16)
        q = torch.arange(-24, 24, dtype=torch.int8).reshape(3, 16)
        scales = torch.tensor([0.1, 0.2, 0.3])
        state = {'embed.weight': q, 'embed.weight_scale': scales, 'embed.comfy_quant': torch.tensor(list(json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 16}).encode()), dtype=torch.uint8)}
        remaining, count = import_comfy_quantized_layers(root, state, orig_dtype=torch.float32)
        self.assertEqual(count, 1)
        self.assertFalse(remaining)
        expected = rotate(q.float() * scales[:, None], 16)
        torch.testing.assert_close(root.embed(torch.tensor([2, 0])), expected[[2, 0]])
        self.assertFalse(root.embed.weight.requires_grad)

    def test_lora_and_lokr_stack_strength_toggle_and_removal(self):
        model = torch.nn.Module(); model.projection = torch.nn.Linear(4, 4, bias=False)
        holder = SimpleNamespace(model=model, text_encoder=None)
        a, b = torch.randn(2, 4), torch.randn(4, 2)
        w1, w2 = torch.randn(2, 2), torch.randn(2, 2)
        lora, lokr = self.root / 'lora.safetensors', self.root / 'lokr.safetensors'
        save_file({'projection.lora_A.weight': a, 'projection.lora_B.weight': b}, lora)
        save_file({'projection.lokr_w1': w1, 'projection.lokr_w2': w2}, lokr)
        x = torch.randn(3, 4); baseline = model.projection(x).detach()
        specs = [{'path': str(lora), 'strength': 0.5}, {'path': str(lokr), 'strength': 0.25}]
        stack = LoRAStack(holder).load(specs); stack.apply()
        torch.testing.assert_close(model.projection(x), baseline + 0.5 * (x @ a.T @ b.T) + 0.25 * (x @ torch.kron(w1, w2).T))
        self.assertTrue(stack.set_strengths([dict(s, strength=0) for s in specs]))
        torch.testing.assert_close(model.projection(x), baseline)
        stack.remove(); torch.testing.assert_close(model.projection(x), baseline)

    def test_prequantized_layer_can_restore_plain_weights_for_new_model_loads(self):
        from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer, OstrisLinear
        from toolkit.util.quantize import dequantize_ostris_to_linear
        model = torch.nn.Sequential(torch.nn.Linear(64, 64))
        self.assertTrue(convert_linear_to_ostris(model[0], get_ostris_quantizer('convrot8', kernel='torch')))
        weight = model[0].dequantize_weight().clone()
        bias = model[0].bias.detach().clone()
        self.assertEqual(dequantize_ostris_to_linear(model), 1)
        self.assertNotIsInstance(model[0], OstrisLinear)
        torch.testing.assert_close(model[0].weight, weight)
        torch.testing.assert_close(model[0].bias, bias)

    def test_stream_replay_and_disconnected_subscriber_cleanup(self):
        job = GenerationJob({'arch': 'test'}, {})
        queue = job.subscribe()
        job.emit('start'); job.emit('latent', b'old'); job.emit('latent', b'new')
        job.unsubscribe(queue)
        self.assertEqual(job._subscribers, [])
        job.status = 'done'; job.finish()
        replay = job.subscribe(); chunks = []
        while True:
            frame = replay.get_nowait()
            if frame is END: break
            chunks.extend(frame[i:i + 3] for i in range(0, len(frame), 3))
        decoded = list(FrameReader(iter(chunks)))
        self.assertEqual([header['type'] for header, _ in decoded], ['start', 'latent', 'end'])
        self.assertEqual(decoded[1][1], b'new')

    def test_generation_validation_rejects_unsafe_paths_and_nonfinite_controls(self):
        for sample in ({'output_path': '../escape'}, {'output_folder': '/tmp'}, {'output_ext': '../txt'}, {'width': float('inf')}, {'duration': -1}):
            with self.subTest(sample=sample), self.assertRaises(ValueError): validate_generation({'model': {'arch': 'flux'}, 'sample': sample})
        for body in ([], {'model': {'arch': 'flux'}, 'sample': []}, {'model': {'arch': 'flux', 'loras': [{'path': 'x', 'strength': float('nan')}]}}):
            with self.assertRaises(ValueError): validate_generation(body)
        self.assertEqual(validate_generation({'model': {'arch': 'qwen25_omni'}, 'sample': {'prompt': 'Describe this'}})[1]['prompt'], 'Describe this')

    def test_http_auth_validation_and_output_containment(self):
        from fastapi.testclient import TestClient
        from extensions_built_in.inference_engine.server import create_app
        engine = Engine('cpu', str(self.root / 'outputs'))
        headers = {'X-Engine-Token': 'test-secret'}
        with TestClient(create_app(engine, 'test-secret')) as client:
            self.assertEqual(client.get('/health').status_code, 401)
            self.assertEqual(client.get('/health', headers=headers).status_code, 200)
            self.assertEqual(client.post('/generate', headers=headers, json={'model': {'arch': 'flux'}, 'sample': {'output_path': '../escape'}}).status_code, 400)
            self.assertEqual(client.post('/assets', headers=headers, content=b'').status_code, 400)
            upload = client.post('/assets?name=../../reference.wav', headers=headers, content=b'fixture').json()
            self.assertEqual(Path(upload['path']).parent, Path(engine.assets_folder))
            self.assertEqual(client.get('/outputs/%2E%2E/secret.txt', headers=headers).status_code, 404)
            result = Path(engine.output_folder) / 'result.txt'; result.write_text('result', encoding='utf8')
            self.assertEqual(client.get('/outputs/result.txt', headers=headers).text, 'result')

    def test_holder_unload_cannot_race_generation_and_stop_preserves_queue_reason(self):
        engine = Engine('cpu', str(self.root / 'outputs'))
        entered, release = threading.Event(), threading.Event()
        def load():
            with engine._holder_lock:
                entered.set(); release.wait(5)
        thread = threading.Thread(target=load)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaises(EngineBusy): engine.unload(wait=False)
        finally:
            release.set(); thread.join(5)
        engine.unload(wait=False)
        engine.stop('queued'); engine.stop('stopped')
        self.assertEqual(engine.stop_reason, 'queued')

    def test_text_samples_publish_atomically_and_are_engine_outputs(self):
        config = GenerateImageConfig(prompt='Instruction', output_folder=str(self.root), output_ext='txt')
        config.save_image_atomic('Generated description')
        files = _list_outputs(str(self.root), include_text=True)
        self.assertEqual(len(files), 1)
        self.assertEqual(Path(files[0]).read_text(encoding='utf8'), 'Generated description')
        self.assertEqual(_kind_for_ext('txt'), 'text')
        self.assertEqual(_list_outputs(str(self.root)), [])

    def test_audio_cache_keeps_fp32_inputs_and_integer_dto_metadata(self):
        tokens = torch.tensor([[1, 123456789]], dtype=torch.long)
        encoder = Mock(return_value=DTO(torch.zeros(1, 2, 4), tokens=tokens, kind=torch.tensor([0])))
        owner = SimpleNamespace(sd=SimpleNamespace(torch_dtype=torch.bfloat16, device_torch='cpu', encode_images=encoder), dataset_config=SimpleNamespace(cache_tensors_to_disk=False, do_i2v=False), transform=None)
        item = SimpleNamespace(tensor=torch.rand(2, 16), is_audio_model=True, is_video=False, audio_data=None, path='fixture.wav', load_and_process_image=Mock(), cleanup=Mock())
        state = LatentCachingMixin._encode_latent_for_file_item(owner, item, str(self.root / 'cache'), False)
        self.assertEqual(encoder.call_args.args[0].dtype, torch.float32)
        LatentCachingMixin._assign_latent_state_to_file_items(owner, [item], state)
        self.assertIsInstance(item._encoded_latent, DTO)
        self.assertEqual(item._encoded_latent.get('tokens').dtype, torch.long)
        self.assertEqual(item._encoded_latent.get('tokens').tolist(), [1, 123456789])

    def test_short_and_silent_audio_artwork(self):
        for values in (np.zeros(12, dtype=np.int16), np.arange(12, dtype=np.int16)):
            filename = self.root / 'short.wav'
            with wave.open(str(filename), 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(values.tobytes())
            envelope = load_waveform(str(filename))
            self.assertEqual(envelope.shape, (512,)); self.assertTrue(np.isfinite(envelope).all())
            self.assertEqual(create_artwork(envelope, 300).size, (300, 300))

    def test_prose_caption_is_not_reformatted_without_tag_operations(self):
        from PIL import Image
        image = self.root / 'sample.png'; Image.new('RGB', (64, 64)).save(image)
        caption = 'Jazz,  warm vocals.\n[Lyrics]\nOne line, another line.'
        image.with_suffix('.txt').write_text(caption, encoding='utf8')
        item = FileItemDTO(path=str(image), dataset_config=DatasetConfig(folder_path=str(self.root), resolution=64, caption_dropout_rate=0, shuffle_tokens=False, token_dropout_rate=0))
        item.load_caption(); self.assertEqual(item.caption, caption)


if __name__ == '__main__': unittest.main()
