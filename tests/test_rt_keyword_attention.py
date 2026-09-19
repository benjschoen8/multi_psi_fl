"""Run with: python3 -m unittest discover -s tests -p 'test_rt_keyword_attention.py'."""
import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from label_mapping.rt_protocol import (attn_coord, keyword_attn, run_rt_protocol, unit,
                                       anchor_log_coordinates, fused_rank, mutual,
                                       rank_bounds, verified_candidates, global_table)


class KeywordAttentionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)
        self.K = unit(self.rng.normal(size=(24, 8)))
        self.c = {
            'summ': {a: np.zeros(8) for a in range(3)},
            'count': {a: 20 for a in range(3)},
            'keywords': {a: ['digit', str(a)] for a in range(3)},
            'keyword_vecs': {a: self.K[[0, a + 1]] for a in range(3)},
            'anchor_vecs': self.K,
        }

    def test_identity_outweighs_context_and_matches_formula(self):
        alpha, _, weights = keyword_attn(self.c, [0, 1, 2], self.K, .1)
        self.assertTrue(np.all(weights.reshape(3, 2)[:, 1] > weights.reshape(3, 2)[:, 0]))
        raw, _ = attn_coord(self.c['keyword_vecs'][0], self.K, self.K, .1)
        np.testing.assert_allclose(alpha[0], weights[:2] @ raw[:, 0])
        np.testing.assert_allclose(alpha.sum(axis=1), 1)

    def test_rotated_encoder_and_keyword_order_preserve_alignment(self):
        rotation, _ = np.linalg.qr(self.rng.normal(size=(8, 8)))
        c2 = {**self.c, 'anchor_vecs': self.K @ rotation,
              'keyword_vecs': {a: v[::-1] @ rotation for a, v in self.c['keyword_vecs'].items()},
              'keywords': {a: t[::-1] for a, t in self.c['keywords'].items()}}
        table, edges, _ = run_rt_protocol({0: self.c, 1: c2}, None, method='attn', psi='plain', log=lambda *_: None)
        self.assertEqual(len(edges), 3)
        for a in range(3):
            self.assertEqual(table[0, a], table[1, a])
        self.assertEqual(len(set(table.values())), 3)

    def test_ragged_uniform_and_legacy(self):
        c = {'keyword_vecs': {0: self.K[:1], 1: self.K[:3]}}
        alpha, _, weights = keyword_attn(c, [0, 1], self.K, .1)
        np.testing.assert_allclose(weights, [1, 1/3, 1/3, 1/3])
        legacy, _, _ = keyword_attn({'desc_vecs': {0: self.K[0]}}, [0], self.K, .1)
        np.testing.assert_allclose(alpha[:1], legacy)

    def test_single_class_and_duplicate_keywords(self):
        alpha, _, weights = keyword_attn(self.c, [0], self.K, .1)
        np.testing.assert_allclose(weights, [.5, .5])
        duplicated = {**self.c,
                      'keywords': {a: ['digit', str(a), str(a)] for a in range(3)},
                      'keyword_vecs': {a: self.K[[0, a + 1, a + 1]] for a in range(3)}}
        expected, _, _ = keyword_attn(self.c, [0, 1, 2], self.K, .1)
        actual, _, _ = keyword_attn(duplicated, [0, 1, 2], self.K, .1)
        np.testing.assert_allclose(actual, expected)

    def test_log_coordinates_preserve_anchor_span_geometry(self):
        q = unit(self.rng.normal(size=(5, 8)))
        alpha, _ = attn_coord(q, self.K, self.K, .05)
        s = anchor_log_coordinates(alpha[:, 0], self.K)
        np.testing.assert_allclose(s @ s.T, q @ q.T, atol=1e-10)
        # Local embedding dimension can differ; anchor coordinates still agree.
        extended = np.pad(self.K, ((0, 0), (0, 4)))
        np.testing.assert_allclose(s, anchor_log_coordinates(alpha[:, 0], extended), atol=1e-10)

    def test_fusion_recovers_text_rejected_candidate_without_raw_scores(self):
        text = np.array([[10, 2], [9, 8]])
        image = np.array([[9, 1], [1, 9]])
        self.assertEqual(mutual(text), [(0, 0)])
        fused = fused_rank(text, image, 10, 9)
        self.assertEqual(mutual(fused), [(0, 0), (1, 1)])
        self.assertEqual(fused_rank(np.array([[0]]), np.array([[9]]), 10, 9)[0, 0], 0)
        self.assertEqual(fused_rank(np.array([[10]]), np.array([[0]]), 10, 9)[0, 0], 0)
        # Two weak signals are not sufficient evidence of a shared class.
        self.assertEqual(fused_rank(np.array([[9]]), np.array([[8]]), 10, 9)[0, 0], 0)
        self.assertGreater(fused_rank(np.array([[9]]), np.array([[8]]), 10, 9,
                                      confidence='positive')[0, 0], 0)
        np.testing.assert_array_equal(fused.T, fused_rank(text.T, image.T, 10, 9))

    @unittest.skipUnless(importlib.util.find_spec('tenseal'), 'TenSEAL not installed')
    def test_keyword_he_matches_plain(self):
        c = {**self.c, 'summ': {a: self.K[a] for a in range(3)}}
        for method in ('attn', 'attn_filter'):
            kwargs = dict(clients={0: c, 1: c}, P=self.K, method=method,
                          ladder=[.9, .5], log=lambda *_: None)
            plain, plain_edges, _ = run_rt_protocol(psi='plain', **kwargs)
            encrypted, encrypted_edges, diag = run_rt_protocol(psi='he', **kwargs)
            self.assertEqual(encrypted, plain)
            self.assertEqual(encrypted_edges, plain_edges)
            self.assertFalse(diag['main_he_mismatch'].any())
            if method == 'attn_filter':
                self.assertFalse(diag['aff_he_mismatch'].any())

    def test_ground_truth_does_not_influence_fusion(self):
        kwargs = dict(clients={0: self.c, 1: self.c}, P=self.K, method='attn_filter',
                      psi='plain', log=lambda *_: None)
        a, ea, _ = run_rt_protocol(gt=lambda *_: True, **kwargs)
        b, eb, _ = run_rt_protocol(gt=lambda *_: False, **kwargs)
        self.assertEqual((a, ea), (b, eb))

    def test_similarity_bounds_are_conservative(self):
        scores = np.array([[-.4, .5, .79, .8, .95, 1.]])
        ladder = [.9, .8, .5]
        ranks = sum((scores > d).astype(int) for d in ladder)
        low, high = rank_bounds(ranks, ladder)
        self.assertTrue((low <= scores).all())
        self.assertTrue((scores <= high).all())

    def test_hidden_rival_prevents_spurious_confidence(self):
        # Second candidate has no text/image nomination but superior verification.
        t = np.array([[1, 0]])
        i = np.array([[1, 0]])
        v = np.array([[2, 3]])
        accepted, _ = verified_candidates(t, i, v, [.4, .9], [.99], [.9, .95, .99])
        self.assertFalse(accepted.any())

    def test_weak_text_rescue_requires_image_margin(self):
        accepted, _ = verified_candidates(np.array([[1]]), np.array([[1]]), np.array([[3]]),
                                           [.4, .9], [.99], [.9, .95, .99])
        self.assertTrue(accepted[0, 0])
        # Strong text cannot override an image verification score below floor.
        accepted, _ = verified_candidates(np.array([[2]]), np.array([[1]]), np.array([[0]]),
                                           [.4, .9], [.99], [.9, .95, .99])
        self.assertFalse(accepted.any())

    def test_complete_link_blocks_transitive_bridge(self):
        edges = [(0, 0, 1, 0, 3), (1, 0, 2, 0, 2)]
        labels = {0: [0], 1: [0], 2: [0]}
        support = {frozenset(((0, 0), (1, 0))), frozenset(((1, 0), (2, 0)))}
        table = global_table(edges, labels, support)
        self.assertEqual(table[0, 0], table[1, 0])
        self.assertNotEqual(table[0, 0], table[2, 0])
        support.add(frozenset(((0, 0), (2, 0))))
        self.assertEqual(len(set(global_table(edges, labels, support).values())), 1)

    def test_invalid_inputs(self):
        for tau in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                run_rt_protocol({0: self.c}, None, method='attn', psi='plain', tau=tau)
        with self.assertRaises(ValueError):
            attn_coord(np.empty((0, 8)), self.K, self.K, .1)
        with self.assertRaises(ValueError):
            keyword_attn({**self.c, 'keywords': {0: ['zero']}}, [0], self.K, .1)

    def test_harness_encodes_separate_multilingual_keywords(self):
        # Load only metadata helpers, avoiding torchvision/dataset imports.
        tree = ast.parse(Path('test_mnist_split_new.py').read_text())
        keep = [n for n in tree.body if
                isinstance(n, ast.FunctionDef) and n.name in ('describe', 'encode_keywords') or
                isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LANGS' for t in n.targets)]
        ns = {'np': np}
        exec(compile(ast.Module(body=keep, type_ignores=[]), '<helpers>', 'exec'), ns)
        descriptions = [ns['describe'](n, 'long') for n in ('zero', '零', 'cero')]
        self.assertEqual(descriptions, [['digit', 'zero'], ['數字', '零'], ['dígito', 'cero']])
        calls = []
        def encode(texts):
            calls.append(texts)
            return np.eye(len(texts))
        result = ns['encode_keywords'](encode, descriptions)
        self.assertEqual(calls, [['digit', 'zero', '數字', '零', 'dígito', 'cero']])
        self.assertEqual([v.shape for v in result], [(2, 6)] * 3)


if __name__ == '__main__':
    unittest.main()
