"""Header sizing must agree with the default EXIF-transposed image loader."""
import io
import struct
import unittest
import zlib
from PIL import Image
from toolkit.image_utils import get_image_size_from_bytesio, _exif_orientation, UnknownImageFormat
from toolkit.image_io import open_static_image_from_bytes


def exif(orientation, endian='<'):
    return (b'II' if endian == '<' else b'MM') + struct.pack(endian + 'HIH', 42, 8, 1) + struct.pack(endian + 'HHIHHI', 274, 3, 1, orientation, 0, 0)


def png_chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))


class FastOrientationTests(unittest.TestCase):
    def size(self, data):
        return get_image_size_from_bytesio(io.BytesIO(data), len(data))

    def test_all_orientations_and_default_loader_equivalence(self):
        for fmt in ('JPEG', 'PNG', 'WEBP'):
            for orientation in range(1, 9):
                for endian in ('<', '>'):
                    with self.subTest(fmt=fmt, orientation=orientation, endian=endian):
                        target = io.BytesIO()
                        Image.new('RGB', (40, 24)).save(target, format=fmt, exif=b'Exif\0\0' + exif(orientation, endian))
                        data = target.getvalue()
                        expected = (24, 40) if orientation >= 5 else (40, 24)
                        self.assertEqual(self.size(data), expected)
                        self.assertEqual(self.size(data), open_static_image_from_bytes(data).size)

    def test_png_exif_after_idat(self):
        target = io.BytesIO()
        Image.new('RGB', (40, 24)).save(target, format='PNG')
        data = target.getvalue()
        data = data[:-12] + png_chunk(b'eXIf', exif(6)) + data[-12:]
        self.assertEqual(self.size(data), (24, 40))
        self.assertEqual(self.size(data), open_static_image_from_bytes(data).size)

    def test_webp_lossy_lossless_extended_and_padding(self):
        for mode, lossless in (('RGB', False), ('RGB', True), ('RGBA', True)):
            target = io.BytesIO()
            Image.new(mode, (40, 24)).save(target, format='WEBP', lossless=lossless)
            data = target.getvalue()
            self.assertEqual(self.size(data), (40, 24))
            chunks = data[12:] + b'JUNK' + struct.pack('<I', 1) + b'x\0' + b'EXIF' + struct.pack('<I', len(exif(8))) + exif(8)
            wrapped = b'RIFF' + struct.pack('<I', len(chunks) + 4) + b'WEBP' + chunks
            self.assertEqual(self.size(wrapped), (24, 40))
            with self.assertRaises(UnknownImageFormat):
                self.size(data[:-1])

    def test_malformed_tiff_and_bounded_metadata_reads(self):
        valid = exif(6)
        for end in range(len(valid)):
            self.assertIsNone(_exif_orientation(valid[:end]))
        for data in (b'ZZ' + valid[2:], valid[:2] + b'\0\0' + valid[4:],
                valid[:4] + b'\xff' * 4 + valid[8:], exif(0), exif(9),
                valid[:14] + struct.pack('<I', 2) + valid[18:]):
            self.assertIsNone(_exif_orientation(data))
        class Bounded(io.BytesIO):
            def read(self, size=-1):
                if size < 0 or size > 1024 * 1024:
                    raise AssertionError('unbounded metadata read')
                return super().read(size)
        target = io.BytesIO()
        Image.new('RGB', (40, 24)).save(target, format='PNG')
        data = target.getvalue()[:33] + struct.pack('>I', 0xffffffff) + b'eXIf'
        self.assertEqual(get_image_size_from_bytesio(Bounded(data), len(data)), (40, 24))
        for data in (b'\xff\xd8\xff\xe1\0\0', b'\xff\xd8\xff\xe1\xff\xff', b'\xff\xd8\xff'):
            with self.assertRaises(UnknownImageFormat):
                self.size(data)

    def test_dataset_size_cache_refresh_preserves_content_identity(self):
        import tempfile
        from pathlib import Path
        from toolkit.data_transfer_object.data_loader import FileItemDTO
        from toolkit.config_modules import DatasetConfig
        from toolkit.basic import get_quick_signature_string
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rotated.jpg'
            Image.new('RGB', (40, 24)).save(path, exif=b'Exif\0\0' + exif(6))
            signature = get_quick_signature_string(str(path))
            database = {'rotated.jpg': (40, 24, signature)}
            fast = FileItemDTO(path=str(path), dataset_config=DatasetConfig(fast_image_size=True), size_database=database)
            self.assertEqual((fast.width, fast.height), (24, 40))
            self.assertEqual(fast.file_signature, signature)
            default = FileItemDTO(path=str(path), dataset_config=DatasetConfig(), size_database=database)
            self.assertEqual((default.width, default.height), (24, 40))
            self.assertEqual(default.file_signature, signature)
