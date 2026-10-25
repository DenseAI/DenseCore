"""GGUF conversion integrity regressions with small synthetic checkpoints."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import convert_julia_gguf as converter


class JuliaConversionTest(unittest.TestCase):
    def checkpoint(self, root, header, payload=b'\0'*16):
        raw=json.dumps(header).encode()
        path=root/'model.safetensors'
        path.write_bytes(struct.pack('<Q',len(raw))+raw+payload)
        return path

    def test_accepts_exact_f32_payload(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(converter,'expected_shapes',return_value={'weight':[2,2]}):
            path=self.checkpoint(Path(tmp),{'weight':{'dtype':'F32','shape':[2,2],'data_offsets':[0,16]}})
            tensors,start=converter.checkpoint_header(path)
            self.assertEqual(tensors['weight']['shape'],[2,2])
            self.assertEqual(path.read_bytes()[start:],b'\0'*16)

    def test_rejects_shape_dtype_and_tensor_substitution(self):
        for name,dtype,shape in [('weight','F32',[4]),('weight','F16',[2,2]),('other','F32',[2,2])]:
            with self.subTest(name=name,dtype=dtype,shape=shape), tempfile.TemporaryDirectory() as tmp, patch.object(converter,'expected_shapes',return_value={'weight':[2,2]}):
                path=self.checkpoint(Path(tmp),{name:{'dtype':dtype,'shape':shape,'data_offsets':[0,16]}})
                with self.assertRaises(ValueError):converter.checkpoint_header(path)

    def test_rejects_overlap_gap_and_truncation(self):
        for offsets,payload in [([8,24],b'\0'*24),([0,16],b'\0'*12),([0,20],b'\0'*20)]:
            with self.subTest(offsets=offsets,length=len(payload)), tempfile.TemporaryDirectory() as tmp, patch.object(converter,'expected_shapes',return_value={'weight':[2,2]}):
                path=self.checkpoint(Path(tmp),{'weight':{'dtype':'F32','shape':[2,2],'data_offsets':offsets}},payload)
                with self.assertRaises(ValueError):converter.checkpoint_header(path)
        with tempfile.TemporaryDirectory() as tmp, patch.object(converter,'expected_shapes',return_value={'one':[2],'two':[2]}):
            path=self.checkpoint(Path(tmp),{'one':{'dtype':'F32','shape':[2],'data_offsets':[0,8]},'two':{'dtype':'F32','shape':[2],'data_offsets':[4,12]}},b'\0'*12)
            with self.assertRaisesRegex(ValueError,'overlapping'):converter.checkpoint_header(path)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(converter,'verify',return_value=True):
            root=Path(tmp);destination=root/'existing.gguf';destination.write_bytes(b'user-owned')
            with self.assertRaisesRegex(ValueError,'refusing to replace'):
                converter.convert(root,destination,root/'not-read.json')
            self.assertEqual(destination.read_bytes(),b'user-owned')

    def test_modified_rope_buffer_is_not_accepted_as_same_provenance(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(converter,'verify',return_value=True):
            root=Path(tmp)
            rope=root/'rope.json'
            rope.write_text(json.dumps({'source_revision':converter.REVISION,
                'encoder_config_sha256':converter.FILES['encoder/config.json'][1],
                'frequencies':{'rope.freq_full':[1.0]*32,'rope.freq_sliding':[1.0]*32}}))
            with self.assertRaisesRegex(ValueError,'frequency bytes'):
                converter.convert(root,root/'new.gguf',rope)
            self.assertFalse((root/'new.gguf').exists())

    def test_bad_source_pin_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);destination=root/'new.gguf'
            with self.assertRaisesRegex(ValueError,'pinned input'):
                converter.convert(root,destination,root/'not-read.json')
            self.assertFalse(destination.exists())


if __name__=='__main__':unittest.main()
