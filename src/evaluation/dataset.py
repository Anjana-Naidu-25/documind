"""Test-set helpers plus the hand-written (curated) questions.

Each test-set line (JSON):
  id, type, question, reference_answer,
  expected_urls   - pages that contain the answer (page-level ground truth)
  expected_chunk_ids - exact chunks the question was written from (chunk-level ground truth)
  should_refuse   - True for out-of-scope questions that must get "I couldn't find this"
"""
import json
from pathlib import Path

from src.config import TESTSET_PATH
from src.ingestion.loader import path_to_url


def load_testset(path: Path = TESTSET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_testset(rows: list[dict], path: Path = TESTSET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Questions that need TWO different pages. Reference answers were written from those pages.
_C = "concepts/"
COMPARISON = [
    ("What is the difference between a ConfigMap and a Secret?",
     "ConfigMaps hold non-confidential configuration data. Secrets are intended for confidential data such as passwords or tokens; by default they are stored unencrypted in the API server's data store, so encryption at rest and access controls are recommended.",
     [_C + "configuration/configmap.md", _C + "configuration/secret.md"]),
    ("When should I use a StatefulSet instead of a Deployment?",
     "Use a StatefulSet for applications that need stable, unique network identities and persistent storage, with ordered deployment and scaling. A Deployment suits stateless applications whose Pods are interchangeable.",
     [_C + "workloads/controllers/deployment.md", _C + "workloads/controllers/statefulset.md"]),
    ("How is a CronJob different from a Job?",
     "A Job runs Pods until a specified number of them complete successfully. A CronJob creates Jobs on a repeating schedule.",
     [_C + "workloads/controllers/job.md", _C + "workloads/controllers/cron-jobs.md"]),
    ("What is the difference between a DaemonSet and a Deployment?",
     "A DaemonSet ensures that a copy of a Pod runs on all (or some) nodes. A Deployment manages a chosen number of replicas that are not tied to particular nodes.",
     [_C + "workloads/controllers/daemonset.md", _C + "workloads/controllers/deployment.md"]),
    ("How do taints and tolerations differ from node affinity?",
     "Node affinity is a property of Pods that attracts them to a set of nodes. Taints are the opposite: they let a node repel a set of Pods, unless the Pods have a matching toleration.",
     [_C + "scheduling-eviction/taint-and-toleration.md", _C + "scheduling-eviction/assign-pod-node.md"]),
    ("What is the difference between namespaces and labels?",
     "Namespaces provide a scope for names and divide cluster resources between groups of users or teams. Labels are key/value pairs attached to objects that identify them and let you select subsets of objects.",
     [_C + "overview/working-with-objects/namespaces.md", _C + "overview/working-with-objects/labels.md"]),
    ("How do init containers differ from sidecar containers?",
     "Init containers run to completion, one at a time, before the app containers start. Sidecar containers are init containers with restartPolicy Always, so they keep running alongside the app containers.",
     [_C + "workloads/pods/init-containers.md", _C + "workloads/pods/sidecar-containers.md"]),
    ("What is the difference between an Ingress and a Service?",
     "A Service exposes a set of Pods as a network service. An Ingress manages external HTTP(S) access to Services using routing rules and needs an ingress controller.",
     [_C + "services-networking/ingress.md", _C + "services-networking/service.md"]),
]

# Clearly unrelated to the Kubernetes docs: the system should answer "I couldn't find this".
OUT_OF_SCOPE = [
    "What is the capital of France?",
    "How do I install Docker on Windows?",
    "Write a Python script that scrapes a website.",
    "How do I use Docker Compose to run several containers?",
    "How much does Google Kubernetes Engine cost per month?",
    "Explain how a transformer neural network works.",
    "How do I reset a forgotten Windows password?",
    "What is the best way to learn React?",
    "How do I train a machine learning model with scikit-learn?",
    "How do I configure a PostgreSQL replica with streaming replication?",
]


def curated_rows() -> list[dict]:
    rows = []
    for i, (q, ref, paths) in enumerate(COMPARISON, 1):
        rows.append({"id": f"cmp-{i:02d}", "type": "comparison", "question": q, "reference_answer": ref,
                     "expected_urls": [path_to_url(p) for p in paths], "expected_chunk_ids": [], "should_refuse": False})
    for i, q in enumerate(OUT_OF_SCOPE, 1):
        rows.append({"id": f"oos-{i:02d}", "type": "out_of_scope", "question": q, "reference_answer": "",
                     "expected_urls": [], "expected_chunk_ids": [], "should_refuse": True})
    return rows
