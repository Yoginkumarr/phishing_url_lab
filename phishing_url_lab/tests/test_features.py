import unittest
import pandas as pd
from features import FEATURE_NAMES, extract_lexical_features, registered_domain_from_url


class TestFeatures(unittest.TestCase):
    def test_schema_and_numeric(self):
        record = extract_lexical_features('https://blog.example.org/posts?id=2')
        self.assertEqual(list(record), FEATURE_NAMES)
        self.assertTrue(all(isinstance(v, (float, int)) for v in record.values()))

    def test_www_and_path_are_invariant(self):
        base = extract_lexical_features('https://www.example.com/')
        changed = extract_lexical_features('http://example.com/docs?a=1#foo')
        self.assertEqual(base, changed)

    def test_labelled_domain_group(self):
        self.assertEqual(registered_domain_from_url('https://a.example.co.uk/path'),
                         registered_domain_from_url('http://b.example.co.uk/'))

    def test_reject_non_http_scheme(self):
        with self.assertRaises(ValueError):
            extract_lexical_features('javascript:alert(1)')

    def test_ip(self):
        self.assertEqual(extract_lexical_features('http://192.0.2.7/a')['host_is_ip'], 1)


if __name__ == '__main__':
    unittest.main()
