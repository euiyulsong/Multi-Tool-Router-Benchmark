import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Unit tests do not contact OpenAI. Allow running tests offline even when
# the user has not installed requirements.txt yet.
import sys
import types
try:
    from openai import OpenAI  # noqa: F401
except ImportError:
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = object
    sys.modules["openai"] = fake_module

import benchmark as b


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def json_call(self, system, payload):
        self.calls += 1
        if 'standalone_query' in system:
            return {'standalone_query': 'Find and list the final report'}
        if 'multi-label tool router' in system:
            names = [x['name'] for x in payload['available_tools']]
            return {'tools': [x for x in names if x.endswith('.find') or x.endswith('.ls')]}
        if 'FINAL' in system:
            return {'tool_calls': [
                {'tool': x['name'], 'arguments': {}, 'search_query': ''}
                for x in payload['selected_tools']]}
        names = [x['name'] for x in payload['available_tools']]
        return {'tool_calls': [
            {'tool': x, 'arguments': {}, 'search_query': ''}
            for x in names if x.endswith('.find') or x.endswith('.ls')]}


class FakeEmbed:
    def __init__(self):
        self.calls = 0
    def search(self, query, k):
        self.calls += 1
        return [('GorillaFileSystem.find', 0.8), ('GorillaFileSystem.ls', 0.7)][:k]


class OfflineTests(unittest.TestCase):
    def build_fixture(self, root):
        (root / 'possible_answer').mkdir()
        (root / 'multi_turn_func_doc').mkdir()
        tools = [
            {'name': 'GorillaFileSystem.find', 'description': 'Find a file',
             'parameters': {'properties': {'name': {'type': 'string'}}}},
            {'name': 'GorillaFileSystem.ls', 'description': 'List files',
             'parameters': {'properties': {}}},
        ]
        # Two extra single-only tools: sanity check tool catalog is not
        # derived from the current test label alone.
        others = [
            {'name': 'do_math', 'description': 'Perform arithmetic', 'parameters': {'properties': {}}},
            {'name': 'summarize', 'description': 'Summarize text', 'parameters': {'properties': {}}},
        ]
        one = {'id': 'parallel_multiple_0', 'question': [[{'role': 'user', 'content': 'Find and list files'}]],
               'function': tools + others}
        gt = {'id': one['id'], 'ground_truth': [
            {'GorillaFileSystem.find': {}}, {'GorillaFileSystem.ls': {}}]}
        multi = {'id': 'multi_turn_base_0', 'question': [
            [{'role': 'user', 'content': 'Look at my reports directory'}],
            [{'role': 'user', 'content': 'Find that report and list the files'}]],
            'involved_classes': ['GorillaFileSystem']}
        multi_gt = {'id': multi['id'], 'ground_truth': [[], ['find(name="report")', 'ls()']]}
        for rel, obj in (
            ('BFCL_v4_parallel_multiple.json', one),
            ('possible_answer/BFCL_v4_parallel_multiple.json', gt),
            ('BFCL_v4_multi_turn_base.json', multi),
            ('possible_answer/BFCL_v4_multi_turn_base.json', multi_gt),
        ):
            (root / rel).write_text(json.dumps(obj)+'\n')
        (root / 'multi_turn_func_doc/gorilla_file_system.json').write_text(
            '\n'.join(json.dumps({'name': t['name'].split('.')[-1],
                                  'description': t['description'],
                                  'parameters': t['parameters']}) for t in tools)+'\n')

    def test_all_nine_settings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.build_fixture(root)
            single = b.read_records(root / 'BFCL_v4_parallel_multiple.json')
            catalog = b.catalog_from_sources(single, root / 'multi_turn_func_doc')
            samples, skips = b.create_examples(root, 42, 1, 1, catalog)
            self.assertEqual(len(samples), 2)
            self.assertEqual(len(skips), 0)
            self.assertEqual([x for x in samples if x.split == 'multi'][0].gold,
                             ['GorillaFileSystem.find', 'GorillaFileSystem.ls'])
            llm, emb = FakeLLM(), FakeEmbed()
            config = SimpleNamespace(retrieve_k=2)
            norm_cache = {}
            outputs = []
            for ex in samples:
                modes = ['raw'] if ex.split == 'single' else ['raw', 'standalone']
                for mode in modes:
                    for arch in ('two_stage', 'one_shot', 'hybrid'):
                        result = b.run_one(ex, mode, arch, catalog, llm, emb, config,
                                           norm_cache, root/'standalone_queries.json')
                        result['latency_sec'] = .01
                        result['error'] = ''
                        outputs.append(result)
            self.assertEqual(len(outputs), 9)
            self.assertEqual(len(norm_cache), 1)
            self.assertTrue(all(r['exact'] == 1 for r in outputs))
            self.assertEqual(len(b.aggregate(outputs)), 9)
            self.assertEqual(emb.calls, 3)  # 1 single + multi raw + multi normalized

    def test_invalid_predictions_penalized(self):
        out = b.score(['GorillaFileSystem.find'], ['GorillaFileSystem.find'], ['unknown'])
        self.assertAlmostEqual(out['precision'], .5)
        self.assertFalse(out['exact'])

    def test_json_parser(self):
        self.assertEqual(b.parse_json_object('```json\n{"tools": ["a"]}\n```')['tools'], ['a'])

    def test_multi_class_resolution_precedes_unqualified(self):
        cat = {'find': {}, 'GorillaFileSystem.find': {}}
        self.assertEqual(b.gt_multi_to_name('find', ['GorillaFileSystem'], cat),
                         'GorillaFileSystem.find')

    def test_dataset_download_uses_cache_and_all_twelve_paths(self):
        urls = []
        def fake_urlopen(request, timeout):
            urls.append(request.full_url)
            return io.BytesIO(b'{"id": "fixture", "ground_truth": []}\n')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'data'
            with patch.object(b.urllib.request, 'urlopen', side_effect=fake_urlopen):
                manifest = b.download_bfcl_dataset(root, timeout=1)
                cached = b.download_bfcl_dataset(root, timeout=1)
            self.assertEqual(len(b.BFCL_DOWNLOAD_FILES), 12)
            self.assertEqual(len(manifest), 12)
            self.assertEqual(len(urls), 12)
            self.assertTrue(all((root / p).exists() for p in b.BFCL_DOWNLOAD_FILES))
            self.assertTrue(all(m['action'] == 'downloaded' for m in manifest))
            self.assertTrue(all(m['action'] == 'cached' for m in cached))

    def test_gpt6_luna_chat_completion_parameters(self):
        sdk = MagicMock()
        sdk.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"tools": []}'))])
        args = SimpleNamespace(chat_url='https://api.openai.com/v1',
                               chat_key='fake-test-key', timeout=5,
                               chat_model='gpt-6-luna', temperature=None,
                               reasoning_effort='none', max_tokens=1600)
        with patch.object(b, 'OpenAI', return_value=sdk):
            self.assertEqual(b.LLM(args).json_call('test', {'foo': 'bar'}), {'tools': []})
        kwargs = sdk.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs['model'], 'gpt-6-luna')
        self.assertEqual(kwargs['reasoning_effort'], 'none')
        self.assertEqual(kwargs['max_completion_tokens'], 1600)
        self.assertEqual(kwargs['response_format'], {'type': 'json_object'})
        self.assertNotIn('temperature', kwargs)
        self.assertNotIn('max_tokens', kwargs)


if __name__ == '__main__':
    unittest.main()
