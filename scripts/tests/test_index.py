"""Regression checks for cross-version and cross-date source uniqueness."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
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

    def test_renamed_conference_title_matches_preprint(self):
        left = {'id': 'arxiv:2610.02140', 'title': 'SFT Learns Better Than You Think',
                'title_aliases': ['Make SFT Generalize, Not Forget'], 'url': 'https://arxiv.org/abs/2610.02140'}
        right = {'id': 'conference:123', 'title': 'Make SFT Generalize, Not Forget', 'url': 'https://example.org/paper/123'}
        self.assertTrue(set(index.aliases(left)) & set(index.aliases(right)))

    def ccf_manifest(self):
        evidence = {'verified': True, 'url': 'https://example.org/official'}
        paper = {'id': 'arxiv:2501.00001', 'title': 'Older Preprint Accepted This Year',
                 'url': 'https://arxiv.org/abs/2501.00001', 'published': '2025-01-01', 'type': 'paper',
                 'paper_type': 'main_conference_full_regular', 'acceptance_year': 2026,
                 'ccf_rank': 'A', 'presentation_type': 'spotlight',
                 'official_evidence': {k: dict(evidence) for k in ('ccf_classification', 'acceptance', 'presentation')},
                 'citation_count': {'value': None, 'unavailable_reason': 'Index unavailable'}}
        return {'date': '2026-11-18', 'direction': 'CCF_Papers', 'selection_year': 2026,
                'selection_basis': 'acceptance_year', 'shortfall_reason': 'Only one eligible paper', 'sources': [paper]}

    def test_ccf_uses_acceptance_year_and_keeps_spotlight_distinct(self):
        item = self.ccf_manifest()
        index.validate_manifest(item, Path('/tmp/CCF_Papers/2026-11-18/manifest.json'))
        self.assertEqual(item['sources'][0]['presentation_type'], 'spotlight')
        item['sources'][0]['acceptance_year'] = 2025
        with self.assertRaisesRegex(ValueError, 'acceptance year'):
            index.validate_manifest(item, Path('/tmp/CCF_Papers/2026-11-18/manifest.json'))

    def test_ccf_requires_full_paper_and_verified_official_evidence(self):
        for change, error in [('type', 'full/regular'), ('evidence', 'official acceptance')]:
            item = self.ccf_manifest()
            if change == 'type':
                item['sources'][0]['paper_type'] = 'workshop'
            else:
                item['sources'][0]['official_evidence']['acceptance']['verified'] = False
            with self.assertRaisesRegex(ValueError, error):
                index.validate_manifest(item, Path('/tmp/CCF_Papers/2026-11-18/manifest.json'))

    def test_ccf_rejects_unattributed_citation_counts(self):
        item = self.ccf_manifest()
        item['sources'][0]['citation_count'] = {'value': 12}
        with self.assertRaisesRegex(ValueError, 'Citation count'):
            index.validate_manifest(item, Path('/tmp/CCF_Papers/2026-11-18/manifest.json'))

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


class NavigationTests(unittest.TestCase):
    def add_report(self, root, direction, day):
        folder = root / direction / day.isoformat()
        folder.mkdir(parents=True)
        key = f'{direction}-{day}'
        source = {
            'id': key, 'title': f'Paper {key}', 'url': f'https://example.org/{key}',
            'published': day.isoformat(),
        }
        manifest = {
            'date': day.isoformat(), 'direction': direction,
            'topic': f'Topic for {day}', 'sources': [source],
        }
        (folder / 'manifest.json').write_text(json.dumps(manifest))
        # Simulate a sparse cloud checkout with existing reports preserved remotely.
        return {f'{direction}/{day}/report.md', f'{direction}/{day}/report.pdf'}

    def test_recent_navigation_stays_small_and_archive_keeps_all_reports(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            start = date(2026, 12, 25)
            days = [start + timedelta(days=offset) for offset in range(8)]
            remote = self.add_report(root, 'Area_B', start)
            for day in reversed(days):
                remote.update(self.add_report(root, 'Area_A', day))
            index.update(root, remote)
            home = (root / 'README.md').read_text()
            direction = (root / 'Area_A/README.md').read_text()
            self.assertEqual(home.count('/report.md)'), 2)
            self.assertIn(f'Area_A/{days[-1]}/report.md', home)
            self.assertNotIn('Paper Area_A', home)
            self.assertEqual(direction.count('/report.md)'), index.RECENT_REPORTS)
            self.assertNotIn(f'{days[0]}/report.md', direction)
            previous = root / 'Area_A/archive/2026-12.md'
            old_month = previous.read_text()
            self.assertIn(f'Paper Area_A-{days[0]}', old_month)
            self.assertIn(f'../{days[0]}/report.pdf', old_month)
            self.assertIn('2027-01.md', (root / 'Area_A/archive/README.md').read_text())

            remote.update(self.add_report(root, 'Area_A', date(2027, 1, 2)))
            with self.assertRaisesRegex(SystemExit, 'refresh'):
                index.update(root, remote, check=True)
            self.assertEqual((root / 'README.md').read_text(), home)
            index.update(root, remote)
            index.update(root, remote, check=True)
            self.assertEqual(previous.read_text(), old_month)
            self.assertEqual(len((root / 'README.md').read_text().splitlines()), len(home.splitlines()))
            self.assertEqual(len((root / 'Area_A/README.md').read_text().splitlines()), len(direction.splitlines()))
            self.assertIn('Paper Area_A-2027-01-02', (root / 'Area_A/archive/2027-01.md').read_text())


if __name__ == '__main__':
    unittest.main()
