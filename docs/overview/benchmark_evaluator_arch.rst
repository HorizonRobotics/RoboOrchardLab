.. _overview_benchmark_evaluator_arch:

Benchmark Evaluator Architecture
================================

Benchmark evaluation runs a policy over a complete benchmark suite and returns
aggregate metrics with per-episode records. The benchmark layer is separate
from :py:class:`~robo_orchard_lab.policy.evaluator.base.PolicyEvaluator`,
which executes one prepared episode at a time.

Most users should start from a concrete domain evaluator, such as
:py:class:`~robo_orchard_lab.policy.evaluator.benchmark.robotwin.RoboTwinBenchmarkEvaluator`.
The lower-level driver and backend concepts are mainly for implementing or
customizing benchmarks.

Core Concepts
-------------

The benchmark API uses three concepts with different audiences:

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Concept
     - Audience
     - Responsibility
   * - :py:class:`~robo_orchard_lab.policy.evaluator.benchmark.BenchmarkEvaluator`
     - Benchmark users
     - Accepts one policy or policy config, runs a complete benchmark, and
       returns :py:class:`~robo_orchard_lab.policy.evaluator.benchmark.BenchmarkResult`.
   * - :py:class:`~robo_orchard_lab.policy.evaluator.benchmark.BenchmarkDriver`
     - Benchmark implementers
     - Owns domain rules such as episode identity, readiness, retry policy,
       seed or offset advancement, artifact paths, and aggregate metrics.
   * - Benchmark backend
     - Framework or advanced implementers
     - Owns worker capacity, prepare/evaluate scheduling, worker lifecycle,
       and cleanup. The remote backend also owns timeout handling and stale
       completion handling.

Layer Boundaries
----------------

The single-episode evaluator, benchmark backend, and domain driver deliberately
own different parts of the system:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Layer
     - Owns
     - Does not own
   * - ``PolicyEvaluator`` / ``PolicyEvaluatorRemote``
     - Environment reset, policy reset, one rollout, metric runtime, remote
       transport, and evaluator-owned cleanup.
     - Multi-episode benchmark orchestration, retry, and aggregation.
   * - ``LocalBenchmarkBackend`` / ``RemoteBenchmarkBackend``
     - Worker capacity, prepare/evaluate pipeline, worker lifecycle, and
       cleanup. The remote backend additionally owns timeout boundaries,
       worker replacement, and stale callback handling.
     - Domain readiness, retry policy, metric aggregation, seed or offset
       semantics, and artifact naming.
   * - ``BenchmarkDriver``
     - Logical episode generation, readiness, retry, attempt request creation,
       seed or offset advancement, final records, and aggregate metrics.
     - Backend scheduling and worker lifecycle.
   * - Domain ``BenchmarkEvaluator``
     - User-facing configuration, backend construction, and compatibility
       entry points.
     - Backend scheduler internals and single-episode rollout mechanics.

Execution Flow
--------------

A typical benchmark evaluator wires a domain driver into a generic backend:

.. code-block:: text

   user policy
       |
       v
   domain BenchmarkEvaluator
       |  creates
       |-- BenchmarkDriver        # domain readiness, retry, metrics
       |-- BenchmarkBackend       # workers and prepare/evaluate scheduling
       |
       v
   BenchmarkBackend.run(driver)
       |
       |-- ask driver for ready jobs
       |-- reserve worker capacity
       |-- prepare worker runtime and reset env
       |-- evaluate rollout from the prepared env start
       |-- report terminal events back to the driver
       |
       v
   BenchmarkResult

The prepare stage exists for benchmarks that must reset the environment before
rollout. It can capture reset metadata such as the actual seed or offset. The
evaluate stage then consumes the prepared start and does not reset the
environment a second time.

Retry Ownership
---------------

Retry policy belongs in the domain driver rather than the generic backend.
Retry decisions often depend on logical episode identity, task grouping, seed
or offset advancement, failure records, and aggregate metric semantics. A
generic backend can reliably report prepare and evaluate failures, replace
failed workers, and continue scheduling, but it should not decide which domain
attempt should be retried next.

For example, RoboTwin offset seeds are assigned during prepare. After a
successful reset, the next offset is based on the actual reset info. Prepare
failures do not advance the seed frontier; once retries are exhausted, the
next logical episode still starts from the current frontier. Evaluate failures
retry from the prepared reset context. These rules are RoboTwin-specific and
therefore stay in ``RoboTwinBenchmarkDriver``.

Adding A Domain Benchmark
-------------------------

New domain benchmarks should usually live under
``robo_orchard_lab/policy/evaluator/benchmark/``:

* Add ``<domain>.py`` for the domain driver, evaluator, and config.
* Implement a ``BenchmarkDriver`` when the domain has its own episode,
  readiness, retry, seed, artifact, or metric policy.
* Reuse ``LocalBenchmarkBackend`` when the domain needs a current-process
  single worker for debugging or smoke tests.
* Reuse ``RemoteBenchmarkBackend`` when the domain needs a remote worker pool,
  timeout isolation, and an explicit prepare/reset stage before rollout.
* Keep compatibility wrappers at old import paths only when existing callers
  need them.

RoboTwinBenchmarkEvaluator
--------------------------

RoboTwin is the first domain benchmark implemented on this architecture.
The user-facing entry point is
:py:class:`~robo_orchard_lab.policy.evaluator.benchmark.robotwin.RoboTwinBenchmarkEvaluator`,
configured by
:py:class:`~robo_orchard_lab.policy.evaluator.benchmark.robotwin.RoboTwinBenchmarkEvaluatorCfg`.

Use it when a policy already implements the RoboOrchard policy evaluation
surface and should be evaluated over a complete RoboTwin task suite:

.. code-block:: python

   from robo_orchard_lab.policy.evaluator.benchmark.robotwin import (
       RoboTwinBenchmarkEvaluatorCfg,
   )

   cfg = RoboTwinBenchmarkEvaluatorCfg(task_names=["<robotwin_task>"])
   result = cfg().evaluate(policy)
   success_rate = result.metrics["average_success_rate"]

``RoboTwinBenchmarkDriver`` owns RoboTwin task readiness, offset advancement,
bounded retry, artifact paths, and success-rate aggregation. The selected
backend owns worker lifecycle and prepare/evaluate mechanics. By default,
``RoboTwinBenchmarkEvaluatorCfg`` uses the local single-worker backend.

Remote backend usage:

.. code-block:: python

   from robo_orchard_lab.policy.evaluator.benchmark.robotwin import (
       RoboTwinBenchmarkEvaluatorCfg,
       RoboTwinRemoteBenchmarkBackendCfg,
   )

   cfg = RoboTwinBenchmarkEvaluatorCfg(
       task_names=["<robotwin_task>"],
       backend=RoboTwinRemoteBenchmarkBackendCfg(num_parallel_envs=4),
   )
   result = cfg().evaluate(policy)

Explicit local backend usage:

.. code-block:: python

   from robo_orchard_lab.policy.evaluator.benchmark.robotwin import (
       RoboTwinBenchmarkEvaluatorCfg,
       RoboTwinLocalBenchmarkBackendCfg,
   )

   cfg = RoboTwinBenchmarkEvaluatorCfg(
       task_names=["<robotwin_task>"],
       backend=RoboTwinLocalBenchmarkBackendCfg(),
       fail_fast=True,
   )
   result = cfg().evaluate(policy)

Configuration
^^^^^^^^^^^^^

The user-facing ``RoboTwinBenchmarkEvaluatorCfg`` fields are:

.. list-table::
   :header-rows: 1
   :widths: 26 18 56

   * - Field
     - Default
     - Meaning
   * - ``task_names``
     - Required
     - RoboTwin task names to evaluate. Names are validated through RoboTwin's
       task registry and must be unique. The list order is also the prepare
       scheduling priority: earlier tasks are selected first whenever they
       have ready retry or new-episode work.
   * - ``episode_num``
     - ``100``
     - Number of final logical episodes recorded per task. Episodes that fail
       after all retries still count in this denominator.
   * - ``max_retries``
     - ``3``
     - Retry count after a failed attempt. One logical episode may therefore
       consume up to ``max_retries + 1`` concrete attempts.
   * - ``max_steps``
     - ``1500``
     - Rollout step cap for each concrete attempt.
   * - ``config_type``
     - ``"demo_clean"``
     - RoboTwin task config file family, for example ``demo_clean`` or
       ``demo_randomized``.
   * - ``start_seed``
     - ``0``
     - Caller-facing benchmark start seed. The driver advances runtime coverage
       through reset-returned ``offset_seed`` values.
   * - ``format_datatypes``
     - ``True``
     - Whether RoboTwin formats observations into RoboOrchard typed data
       objects before passing them to the policy.
   * - ``action_type``
     - ``"qpos"``
     - Action representation expected by the RoboTwin env: joint-position
       ``qpos`` actions or end-effector ``ee`` pose actions.
   * - ``backend``
     - ``RoboTwinLocalBenchmarkBackendCfg()``
     - Backend runtime config. Use
       ``RoboTwinLocalBenchmarkBackendCfg()`` for current-process single
       worker debug runs, or ``RoboTwinRemoteBenchmarkBackendCfg(...)`` for
       Ray-backed execution with per-call timeout isolation.
   * - ``fail_fast``
     - ``False``
     - Raise ``BenchmarkAttemptError`` on the first infrastructure failure
       instead of bounded retry and failed episode recording.
   * - ``artifact_root_dir``
     - ``None``
     - Optional root for artifacts such as RoboTwin videos. The driver passes
       a task/config directory, and the env writes videos as
       ``episode_{episode_id}_seed_{actual_seed}.mp4`` after reset resolves
       the actual seed.

The local backend config has no fields. It always runs one current-process
worker and intentionally has no hard reset or rollout timeout.

The remote backend config fields are:

.. list-table::
   :header-rows: 1
   :widths: 26 18 56

   * - Field
     - Default
     - Meaning
   * - ``num_parallel_envs``
     - ``1``
     - Number of remote RoboTwin env workers. Prepare/reset remains
       per-task sequential to avoid offset collisions; prepared rollouts may
       run concurrently.
   * - ``rollout_timeout_s``
     - ``120.0``
     - Per-call timeout for rollout/evaluate calls.
   * - ``reset_timeout_s``
     - ``1200.0``
     - Per-call timeout for setup, env reconfigure, metric reset, env reset,
       and metric snapshot calls.
   * - ``remote_class_config``
     - 8 CPU, 1 GPU, 16 GiB
     - Ray actor resource request for each remote policy evaluator worker.
       Tune this when RoboTwin env or policy resource needs differ.
   * - ``ray_init_config``
     - ``None``
     - Optional Ray initialization config forwarded when the backend creates
       the remote evaluator pool.

``episode_timeout_s`` and ``worker_poll_interval_s`` are not part of the new
config. The benchmark backend uses separate reset and rollout timeouts, and
worker scheduling is event-driven rather than sleep-polling.

Compared With RoboTwin's Official Local Evaluator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RoboTwin's official ``script/eval_policy.py`` is the reference local
single-worker evaluator. It loads a policy through RoboTwin's deploy-policy
module convention, creates one task env in the current process, searches valid
evaluation seeds through the expert-check path, runs 100 accepted episodes
sequentially, and writes task-local text results and optional videos.

``RoboTwinBenchmarkEvaluator`` is a reusable benchmark interface instead of a
single-process evaluation script. The main advantages are:

* **Policy-agnostic entry point**: callers provide any compatible policy or
  policy config. Checkpoint loading and RoboTwin deploy-policy module loading
  stay outside the benchmark runner.
* **Typed benchmark result**: the evaluator returns ``BenchmarkResult`` with
  aggregate metrics, per-episode records, attempts, and failure details instead
  of only relying on local text/video side effects.
* **Bounded retry with benchmark accounting**: failed attempts retry up to
  ``max_retries``; a logical episode that still fails is recorded and counted
  before the task advances.
* **Offset-safe prepare stage**: RoboTwin reset happens before rollout, so the
  driver can use actual reset-returned ``offset_seed`` values and avoid
  allocating conflicting offsets prematurely.
* **Local and remote backends**: local backend keeps the official evaluator's
  current-process single-worker debugging model; remote backend adds parallel
  worker capacity while preserving RoboTwin's per-task prepare ordering.
* **Explicit timeout and cleanup boundaries**: remote reset/setup and rollout
  have separate timeouts, workers are closed through backend lifecycle
  management, and stale completions are ignored after worker replacement.
* **Fast unit-test seam**: the backend and driver can be tested with fake
  remote evaluators instead of launching the RoboTwin simulator in normal unit
  tests.
