import unittest

from src.utils import build_label, parse_source


class UtilsTests(unittest.TestCase):
    def test_webcam_source_becomes_integer(self):
        self.assertEqual(parse_source("0"), 0)

    def test_video_path_remains_string(self):
        self.assertEqual(parse_source("videos/test.mp4"), "videos/test.mp4")

    def test_label_with_tracking_id(self):
        label = build_label(7, "person", 0.91)
        self.assertIn("ID 7", label)
        self.assertIn("person", label)
        self.assertIn("0.91", label)

    def test_label_without_tracking_id(self):
        label = build_label(None, "car", 0.75)
        self.assertIn("ID ?", label)


if __name__ == "__main__":
    unittest.main()
