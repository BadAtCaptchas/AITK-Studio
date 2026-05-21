import sqlite3
import tempfile
import unittest
from pathlib import Path

from toolkit.logging_aitk import MongoUILogger, UILogger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeMetricsCollection:
    def __init__(self, docs):
        self.docs = list(docs)

    def delete_many(self, query):
        job_id = query["job_id"]
        max_step = query["step"]["$gt"]
        self.docs = [
            doc
            for doc in self.docs
            if not (doc.get("job_id") == job_id and doc.get("step", -1) > max_step)
        ]

    def aggregate(self, _pipeline):
        grouped = {}
        for doc in self.docs:
            if doc.get("job_id") != "job-1":
                continue
            key = doc["key"]
            current = grouped.setdefault(
                key,
                {"_id": key, "first_seen_step": doc["step"], "last_seen_step": doc["step"]},
            )
            current["first_seen_step"] = min(current["first_seen_step"], doc["step"])
            current["last_seen_step"] = max(current["last_seen_step"], doc["step"])
        return list(grouped.values())


class FakeMetricKeysCollection:
    def __init__(self, docs):
        self.docs = list(docs)

    def replace_one(self, query, replacement, upsert=False):
        for index, doc in enumerate(self.docs):
            if all(doc.get(key) == value for key, value in query.items()):
                self.docs[index] = replacement
                return
        if upsert:
            self.docs.append(replacement)

    def delete_many(self, query):
        job_id = query["job_id"]
        keep_keys = set(query.get("key", {}).get("$nin", []))
        if keep_keys:
            self.docs = [
                doc
                for doc in self.docs
                if doc.get("job_id") != job_id or doc.get("key") in keep_keys
            ]
        else:
            self.docs = [doc for doc in self.docs if doc.get("job_id") != job_id]


class SelectiveUpstreamPortsTest(unittest.TestCase):
    def test_sqlite_ui_logger_prunes_future_steps_on_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "loss_log.db")

            logger = UILogger(log_path, flush_every_n=1, flush_every_secs=999)
            for step in range(4):
                logger.log({"loss": step})
                logger.commit(step=step)
            logger.finish()

            resumed = UILogger(log_path, flush_every_n=1, flush_every_secs=999)
            resumed.log({"loss": 100})
            resumed.commit(step=1)
            resumed.finish()

            con = sqlite3.connect(log_path)
            try:
                steps = [row[0] for row in con.execute("SELECT step FROM steps ORDER BY step")]
                metrics = list(con.execute("SELECT step, key, value_real FROM metrics ORDER BY step"))
                keys = list(con.execute("SELECT key, first_seen_step, last_seen_step FROM metric_keys"))
            finally:
                con.close()

            self.assertEqual(steps, [0, 1])
            self.assertEqual(metrics, [(0, "loss", 0.0), (1, "loss", 100.0)])
            self.assertEqual(keys, [("loss", 0, 1)])

    def test_mongo_ui_logger_prunes_future_steps_on_resume(self):
        logger = MongoUILogger("unused")
        logger.job_id = "job-1"
        logger._metrics = FakeMetricsCollection(
            [
                {"job_id": "job-1", "step": 0, "key": "loss"},
                {"job_id": "job-1", "step": 1, "key": "loss"},
                {"job_id": "job-1", "step": 3, "key": "loss"},
                {"job_id": "job-1", "step": 2, "key": "lr"},
                {"job_id": "job-2", "step": 9, "key": "loss"},
            ]
        )
        logger._metric_keys = FakeMetricKeysCollection(
            [
                {"job_id": "job-1", "key": "loss", "first_seen_step": 0, "last_seen_step": 3},
                {"job_id": "job-1", "key": "lr", "first_seen_step": 2, "last_seen_step": 2},
                {"job_id": "job-2", "key": "loss", "first_seen_step": 9, "last_seen_step": 9},
            ]
        )

        logger._prune_future_steps(1)

        self.assertEqual(
            logger._metrics.docs,
            [
                {"job_id": "job-1", "step": 0, "key": "loss"},
                {"job_id": "job-1", "step": 1, "key": "loss"},
                {"job_id": "job-2", "step": 9, "key": "loss"},
            ],
        )
        self.assertIn(
            {"job_id": "job-1", "key": "loss", "first_seen_step": 0, "last_seen_step": 1},
            logger._metric_keys.docs,
        )
        self.assertNotIn(
            {"job_id": "job-1", "key": "lr", "first_seen_step": 2, "last_seen_step": 2},
            logger._metric_keys.docs,
        )

    def test_dfe_gradient_checkpointing_enable_is_supported(self):
        source = (PROJECT_ROOT / "extensions_built_in" / "sd_trainer" / "SDTrainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("gradient_checkpointing_enable", source)
        self.assertIn("self.dfe.model.gradient_checkpointing_enable()", source)

    def test_sapiens2_mask_is_routed_as_mask(self):
        config_source = (PROJECT_ROOT / "toolkit" / "config_modules.py").read_text(encoding="utf-8")
        control_source = (PROJECT_ROOT / "toolkit" / "control_generator.py").read_text(encoding="utf-8")
        dataloader_source = (PROJECT_ROOT / "toolkit" / "dataloader_mixins.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("'sapiens2_mask'", config_source)
        self.assertIn("Sapiens2Matting.from_pretrained", control_source)
        self.assertIn("control_type == 'mask' or control_type == 'sapiens2_mask'", dataloader_source)


if __name__ == "__main__":
    unittest.main()
