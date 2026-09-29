"""Synthetic negative tests. Never invoke training, search, or full validation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import validate_gold_s4_secondary_improvement_run_v1_1 as repair
from training_holdout_guard_v1 import install


class RepairTests(unittest.TestCase):
    def binding(self, path):
        raw = b'print(1)\n'
        binding = dict(path=path, role=repair.ROLES[path], sha256=repair.byte_sha(raw),
                       git_commit='a'*40, git_blob_sha256=repair.byte_sha(raw),
                       git_representation='EXACT_BYTES')
        return binding, path, repair.byte_sha(raw), 'a'*40, raw, raw

    def test_historical_and_current_roles(self):
        for path in repair.ROLES:
            with self.subTest(path=path):
                repair.verify_binding(*self.binding(path))

    def test_filename_only(self):
        args = list(self.binding(repair.HISTORICAL))
        args[0] = {'path': repair.HISTORICAL}
        with self.assertRaises(ValueError):
            repair.verify_binding(*args)

    def test_wrong_hash_both_roles(self):
        for path in (repair.HISTORICAL, 'gold_s4_secondary_improvement_v1.py'):
            args = list(self.binding(path))
            args[0]['sha256'] = '0'*64
            with self.subTest(path=path), self.assertRaises(ValueError):
                repair.verify_binding(*args)

    def test_role_swap(self):
        args = list(self.binding(repair.HISTORICAL))
        args[0]['role'] = 'CURRENT_IMPROVEMENT_EXECUTION'
        with self.assertRaises(ValueError):
            repair.verify_binding(*args)

    def test_missing_binding_fields(self):
        for field in self.binding(repair.HISTORICAL)[0]:
            args = list(self.binding(repair.HISTORICAL))
            del args[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                repair.verify_binding(*args)

    def test_missing_source_binding(self):
        with patch.object(repair, 'read', return_value={'source_bindings': {}}):
            with self.assertRaisesRegex(ValueError, 'missing_role_binding'):
                repair.source_diagnosis(Path('synthetic'))

    def test_real_windows_representation(self):
        manifest = repair.read(repair.ROOT/'training_runs'/repair.TARGET_ID/'manifest.json')
        raw = (repair.ROOT/repair.HISTORICAL).read_bytes()
        blob = raw.replace(b'\r\n', b'\n')
        self.assertNotEqual(raw, blob)
        expected = manifest['source_bindings'][repair.HISTORICAL]
        binding = dict(path=repair.HISTORICAL, role=repair.ROLES[repair.HISTORICAL],
                       sha256=expected, git_commit=manifest['git_commit'],
                       git_blob_sha256=repair.HISTORICAL_BLOB_SHA, git_representation='CRLF_TO_LF')
        repair.verify_binding(binding, repair.HISTORICAL, expected, manifest['git_commit'], raw, blob)
        with self.assertRaises(ValueError):
            repair.verify_binding(binding, repair.HISTORICAL, expected, manifest['git_commit'], raw+b' ', blob)
        with self.assertRaises(ValueError):
            repair.verify_binding(binding, repair.HISTORICAL, expected, manifest['git_commit'], raw, blob+b' ')

    def test_normalization_not_general_bypass(self):
        args = list(self.binding('gold_s4_secondary_improvement_v1.py'))
        args[4] = b'print(1)\r\n'
        args[2] = args[0]['sha256'] = repair.byte_sha(args[4])
        args[0]['git_representation'] = 'CRLF_TO_LF'
        with self.assertRaises(ValueError):
            repair.verify_binding(*args)

    def test_mutation_rejected(self):
        for name in ('model.json', 'candidate_metrics.json', 'predeclared_search_space.json',
                     'selected_candidate.json', 'gemini.py', 'validator.json', 'validator_attempt.json', 'FINALIZED.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/name).write_bytes(b'original')
                hashes = {name: repair.sha(root/name)}
                repair.verify_inventory(root, hashes)
                (root/name).write_bytes(b'mutated')
                with self.assertRaises(ValueError):
                    repair.verify_inventory(root, hashes)

    def test_holdout_and_target_writes_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'repair'
            output.mkdir()
            release = install(root, write_root=output)
            try:
                with self.assertRaises(PermissionError):
                    (root/'future_holdout/gold_s4_v4/no-real-data').read_bytes()
                with self.assertRaises(PermissionError):
                    (root/'validator.json').write_text('overwrite')
                repair.write_new(output/'allowed.json', {'allowed': True})
            finally:
                release()

    def test_additive_pass_requires_full_validation_same_run(self):
        good = dict(overall='PASS', full_validation_completed=True, run_id=repair.TARGET_ID,
                    original_inventory_unchanged=True, source_binding_schema_status='PASS')
        self.assertEqual(repair.amendment_payload(good, 'repair', 'a'*64)['effective_final_status'], 'PASS')
        for key, value in [('overall', 'FAIL'), ('full_validation_completed', False),
                           ('run_id', 'another'), ('original_inventory_unchanged', False),
                           ('source_binding_schema_status', 'FAIL')]:
            bad = {**good, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                repair.amendment_payload(bad, 'repair', 'a'*64)

    def test_exclusive_receipts_preserve_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'validator.json'
            repair.write_new(path, {'overall': 'FAIL'})
            with self.assertRaises(FileExistsError):
                repair.write_new(path, {'overall': 'PASS'})
            self.assertEqual(repair.read(path), {'overall': 'FAIL'})

    def test_training_and_search_blocked(self):
        class Frame:
            pass
        for module, name in [('xgboost.sklearn', 'fit'), ('xgboost.training', 'train'),
                             ('xgboost.core', 'update'), ('sklearn.linear_model', 'fit'),
                             ('gold_s4_secondary_improvement_v1', 'research'),
                             ('gold_manual_s4_secondary_retrain_v1', 'train_folds'),
                             ('gold_s4_secondary_improvement_logic_v1', 'select')]:
            frame = Frame()
            frame.f_globals = {'__name__': module}
            frame.f_code = type('Code', (), {'co_name': name})()
            with self.subTest(module=module, name=name), self.assertRaises(PermissionError):
                repair.reject_training(frame, 'call', None)

    def test_original_run_identity(self):
        with self.assertRaisesRegex(ValueError, 'same_original_run'):
            repair.check_original(repair.ROOT/'training_runs'/'another')


if __name__ == '__main__':
    unittest.main(verbosity=2)
