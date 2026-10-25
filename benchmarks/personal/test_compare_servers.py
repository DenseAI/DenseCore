import unittest
from compare_servers import validate_response

class FinalAnswerGateTest(unittest.TestCase):
    def test_reasoning_match_does_not_pass_missing_final(self):
        with self.assertRaises(ValueError): validate_response('', 'Paris', 'stop', {'completion_tokens': 3}, 'Paris')
        with self.assertRaises(ValueError): validate_response('<think>Paris', '', 'length', {'completion_tokens': 3}, 'Paris')
    def test_truncated_output_fails_even_with_correct_text(self):
        with self.assertRaises(ValueError): validate_response('Paris', '', 'length', {'completion_tokens': 3}, 'Paris')
    def test_only_final_channel_matches(self):
        with self.assertRaises(ValueError): validate_response('<think>Paris</think>London', '', 'stop', {'completion_tokens': 3}, 'Paris')
        self.assertEqual(validate_response('<think>London</think>Paris', '', 'stop', {'completion_tokens': 3}, 'Paris'), 'Paris')
if __name__ == '__main__': unittest.main()
