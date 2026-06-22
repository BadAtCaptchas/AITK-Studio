import unittest

from toolkit.dataloader_mixins import Bucket, BucketsMixin


class DummyBucketDataset(BucketsMixin):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size


class BucketBatchIndicesTest(unittest.TestCase):
    def test_bucket_smaller_than_batch_size_is_padded(self):
        dataset = DummyBucketDataset(batch_size=4)
        bucket = Bucket(512, 512)
        bucket.file_list_idx = [7]
        dataset.buckets = {"512x512": bucket}

        dataset.build_batch_indices()

        self.assertEqual(dataset.batch_indices, [[7, 7, 7, 7]])

    def test_final_partial_bucket_batch_is_padded(self):
        dataset = DummyBucketDataset(batch_size=2)
        bucket = Bucket(512, 512)
        bucket.file_list_idx = [0, 1, 2, 3, 4]
        dataset.buckets = {"512x512": bucket}

        dataset.build_batch_indices()

        self.assertEqual(dataset.batch_indices, [[0, 1], [2, 3], [4, 4]])

    def test_final_partial_bucket_batch_with_multiple_leftovers_is_padded(self):
        dataset = DummyBucketDataset(batch_size=4)
        bucket = Bucket(512, 512)
        bucket.file_list_idx = [0, 1, 2, 3, 4, 5]
        dataset.buckets = {"512x512": bucket}

        dataset.build_batch_indices()

        self.assertEqual(dataset.batch_indices, [[0, 1, 2, 3], [4, 5, 4, 5]])

    def test_empty_bucket_does_not_produce_batch(self):
        dataset = DummyBucketDataset(batch_size=3)
        bucket = Bucket(512, 512)
        dataset.buckets = {"512x512": bucket}

        dataset.build_batch_indices()

        self.assertEqual(dataset.batch_indices, [])

    def test_batch_size_one_is_unchanged(self):
        dataset = DummyBucketDataset(batch_size=1)
        bucket = Bucket(512, 512)
        bucket.file_list_idx = [3, 4]
        dataset.buckets = {"512x512": bucket}

        dataset.build_batch_indices()

        self.assertEqual(dataset.batch_indices, [[3], [4]])


if __name__ == "__main__":
    unittest.main()
