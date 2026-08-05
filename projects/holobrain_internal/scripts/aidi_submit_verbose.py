#!/usr/bin/env python3
"""Submit an AIDI job and print the job id.

`RoboOrchardJob-AIDISubmit aidisdk_job_submit` discards what submit_job
returns (aidi_submit_job.py:141; the print above it is commented out), so a
successful submission produces rc=0 and no output at all -- indistinguishable
from a silent failure. And `aidictl job list` cannot see a freshly created
job: one confirmed by `aidictl job status` as Queuing was invisible to the
list under every filter tried (-n, -p, -c, -u all, -d 1). So "no output, and
not in the list" is not evidence that nothing was submitted, and retrying on
that basis creates duplicates that cannot then be enumerated to stop.

This does the same two calls and prints the id, which `aidictl job status`
does answer correctly.

    aidi_submit_verbose.py <job_config.yaml> <queue_name> [job_type]
"""

import sys
from uuid import uuid1

from aidisdk import AIDIClient
from aidisdk.compute import CodePackageConfig, LocalPackageItem, StartUpConfig
from robo_orchard_jobs.job_submit.aidi.aidi_submit_job import (
    AIDISDKSubmitConfig,
)
from robo_orchard_jobs.job_submit.aidi.job_config import AIDIJobParams


def main() -> None:
    path, queue = sys.argv[1], sys.argv[2]
    job_type = sys.argv[3] if len(sys.argv) > 3 else "train"

    cfg = AIDISDKSubmitConfig(
        job_config_path=path, queue_name=queue, job_type=job_type
    )
    jc = AIDIJobParams.from_str(open(path).read(), format="yaml")
    client = AIDIClient()
    package = LocalPackageItem(
        lpath=jc.REQUIRED.UPLOAD_DIR, encrypt_passwd=jc.REQUIRED.JOB_PASSWD
    )
    name = "%s_%s" % (jc.REQUIRED.JOB_NAME, str(uuid1()).replace("-", "_"))
    job = client.single_job.new_job(
        priority=jc.OPTIONAL.PRIORITY,
        job_name=name,
        job_type=job_type,
        project_id=jc.REQUIRED.PROJECT_ID,
        queue_name=queue,
        running_resource=cfg._get_running_resources(jc),
        mount=cfg._get_mount_items(jc),
        startup=StartUpConfig(
            command=jc.REQUIRED.RUN_SCRIPTS, startup_dir=package
        ),
        code_package=CodePackageConfig(
            raw_package=package.set_as_startup_dir()
        ),
    )
    task = client.single_job.submit_job(job)
    job_id = getattr(task, "job_id", None)
    if not job_id:
        raise SystemExit("SUBMIT RETURNED NO JOB ID: %r" % (task,))
    print("SUBMITTED %s" % job_id)
    print("NAME      %s" % name)
    print("CHECK     aidictl job status %s" % job_id)


if __name__ == "__main__":
    main()
