"""Regression checks for cross-version and cross-date source uniqueness."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import update_index as index


class AliasTests(unittest.TestCase):
    def test_arxiv_versions_pdf_and_html_are_one_paper(self):
        values = ['arxiv:2611.01234v2', 'https://arxiv.org/pdf/2611.01234v3.pdf', 'https://arxiv.org/html/2611.01234v1']
        self.assertEqual({index.arxiv_id(v) for v in values}, {'2611.01234'})

    def test_doi_and_tracking_urls(self):
        self.assertEqual(index.doi_id('https://doi.org/10.1234/ABC?utm_source=x'), '10.1234/abc')
        self.assertEqual(index.canonical_url('http://Example.org/paper/?utm_source=x#intro'), 'https://example.org/paper')

    def test_normalized_title_crosses_venues(self):
        self.assertEqual(index.normalized_title('Vision-OPD: A Study'), index.normalized_title('VISION OPD — A Study'))

    def test_shared_doi_detects_different_arxiv_ids(self):
        left = {'id': 'arxiv:2611.00001', 'title': 'First title', 'url': 'https://arxiv.org/abs/2611.00001', 'doi': '10.1000/ABC'}
        right = {'id': 'arxiv:2611.00002v2', 'title': 'Different title', 'url': 'https://arxiv.org/pdf/2611.00002v2', 'doi': 'https://doi.org/10.1000/abc'}
        self.assertIn('doi:10.1000/abc', set(index.aliases(left)) & set(index.aliases(right)))

    def test_shortfall_and_window_policy(self):
        item = {'date': '2026-11-17', 'direction': 'VLM', 'window_start': '2026-10-17', 'window_end': '2026-11-17', 'sources': []}
        path = Path('/tmp/VLM/2026-11-17/manifest.json')
        with self.assertRaisesRegex(ValueError, 'shortfall'):
            index.validate_manifest(item, path)
        item['shortfall_reason'] = 'No new qualifying papers after deduplication'
        index.validate_manifest(item, path)
        item['sources'] = [{'id': 'arxiv:2601.00001', 'title': 'Old work', 'url': 'https://arxiv.org/abs/2601.00001', 'type': 'paper', 'published': '2026-01-01'}]
        with self.assertRaisesRegex(ValueError, 'outside window'):
            index.validate_manifest(item, path)

    def test_update_rejects_duplicate_before_writing(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for day, sid in [('2026-11-17', 'arxiv:2611.01234v1'), ('2026-11-18', 'arxiv:2611.01234v2')]:
                folder = root / 'VLM' / day
                folder.mkdir(parents=True)
                month_start = day.replace('11-', '10-')
                data = {'date': day, 'direction': 'VLM', 'topic': 'Test', 'window_start': month_start, 'window_end': day, 'shortfall_reason': 'Only one new paper', 'sources': [{'id': sid, 'title': 'Same Paper', 'url': 'https://arxiv.org/abs/' + sid[6:], 'type': 'paper', 'published': '2026-11-01'}]}
                (folder / 'manifest.json').write_text(json.dumps(data))
                (folder / 'report.md').write_text('test')
                (folder / 'report.pdf').write_bytes(b'%PDF-test')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                index.update(root)
            self.assertFalse((root / 'README.md').exists())


if __name__ == '__main__':
    unittest.main()
