# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import logging
import time
from enum import Enum

# ruff: noqa: E501, D205, N806


class ReturnCode(int, Enum):
    SUCCESS = 0
    FAILURE = 1
    TIMEOUT = 2
    EXCEPTION = 3


def process_job(
    client,
    gpu_client,
    job_id,
    robot_id,
    image_size,
    image_type,
    action_type,
    duration,
    prompt=None,
    max_wait=600,
    task_name=None,
):
    """Handles the processing of a single job, including status checking, robot starting,
    state polling, inference, and action posting.

    Args:
        client: An instance of the InterfaceClient for interacting with the job system.
        gpu_client: An inference client that wraps the policy/model for decision-making.
        job_id (str): The unique identifier for the job to process.
        robot_id (str): The unique identifier for the robot associated with the job.
        image_size (list): The size of images to request from the robot (e.g., [224, 224]).
        image_type (list): The types of images to request (e.g., ["cam_left_wrist", "cam_right_wrist", "cam_high"]).
        action_type (str): The type of action to perform (e.g., "joint").
        duration (float): The duration for each action command.
        prompt (str, optional): Task prompt associated with the run.
        max_wait (int, optional): Maximum time to wait for job completion in seconds. Defaults to 600.

    Notes:
        - This function should not be modified by users.
        - It performs the main job loop for a single job, including error handling and logging.
        - For more details about parameters, see README.md.
    """
    try:
        device, status = client.get_job_status(job_id)
        logging.info(f"Processing job_id: {job_id}, status: {status}")
        if status == "ready":
            client.update_job_info(job_id, robot_id)
            r = client.start_robot(job_id)
            logging.info(f"Started robot: {r.content}")
            if r.status_code == 200:
                wait_result = client.wait_for_robot_running(job_id)
                if wait_result != ReturnCode.SUCCESS:
                    logging.warning(
                        f"Job {job_id} failed to reach running state: {wait_result}"
                    )
                    return

                start_time = time.time()
                while True:
                    device, status = client.get_job_status(job_id)
                    if status != "running":
                        logging.info(
                            f"Job {job_id} left running state: {status}"
                        )
                        break
                    state = client.get_state(
                        image_size, image_type, action_type
                    )
                    if not state:
                        time.sleep(0.5)
                        continue
                    if (
                        state["state"] != "normal"
                        or state["pending_actions"] != 0
                    ):
                        time.sleep(0.5)
                        continue
                    logging.info(
                        "get_robot_state time: %.2f",
                        time.time() - state["timestamp"],
                    )

                    state["job_id"] = job_id
                    state["task_name"] = task_name
                    result = gpu_client.infer(state, prompt=prompt)
                    logging.info(f"Inference result: {result}")
                    client.post_actions(result, duration, action_type)
                    if time.time() - start_time > max_wait:
                        logging.warning(
                            f"Job {job_id} exceeded max wait time."
                        )
                        break
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")


def job_loop(
    client,
    gpu_client,
    submission_id,
    image_size,
    image_type,
    action_type,
    duration,
):
    """Main loop for polling and processing all jobs in a job collection.

    Args:
        client: An instance of the InterfaceClient for interacting with the job system.
        gpu_client: An inference client that wraps the policy/model for decision-making.
        submission_id (str): The unique identifier for the submission to monitor.
        image_size (list): The size of images to request from the robot.
        image_type (list): The types of images to request.
        action_type (str): The type of action to perform.
        duration (float): The duration for each action command.

    Notes:
        - This function repeatedly polls the job collection for active jobs.
        - It processes jobs with status "ready" by calling process_job.
        - The loop exits if no active jobs are found after several consecutive polls.
        - This function should not be modified by users.
        - For more details about parameters, see README.md.
    """
    ACTIVE_STATES = ["assigned", "prepare", "ready", "running"]
    MAX_EMPTY_POLLS = 10
    empty_poll_count = 0
    current_run_id = None
    current_prompt = None

    while True:
        job_collections = client.get_all_runs(submission_id)
        target_job_collection = None

        if current_run_id is not None:
            target_job_collection = next(
                (
                    job_collection
                    for job_collection in job_collections
                    if job_collection.get("run_id") == current_run_id
                    and job_collection.get("status") in ACTIVE_STATES
                ),
                None,
            )

        if target_job_collection is None:
            for preferred_status in [
                "prepare",
                "ready",
                "running",
                "assigned",
            ]:
                target_job_collection = next(
                    (
                        job_collection
                        for job_collection in job_collections
                        if job_collection.get("status") == preferred_status
                    ),
                    None,
                )
                if target_job_collection is not None:
                    break

        if target_job_collection is None:
            if current_run_id is not None:
                logging.info(
                    f"Run {current_run_id} is no longer active, waiting for the next run..."
                )
                current_run_id = None
                current_prompt = None
            else:
                logging.info(
                    f"No active run found for submission {submission_id}, waiting..."
                )
            empty_poll_count = 0
            time.sleep(2)
            continue

        selected_run_id = target_job_collection["run_id"]
        if selected_run_id != current_run_id:
            current_run_id = selected_run_id
            current_prompt = target_job_collection.get("prompt")
            empty_poll_count = 0
            task_name = target_job_collection["task_name"]
            robot_tag = target_job_collection["robotTag"]
            status = target_job_collection["status"]
            logging.info(
                f"job_collection  id: {current_run_id}, task name: {task_name}, prompt: {current_prompt}, robot tag: {robot_tag}, status: {status}"
            )

        job_collection = client.get_all_jobs(current_run_id)
        jobs = job_collection.get("jobs", [])

        has_active_job = False
        exit_code = 0
        for job in jobs:
            status = job["status"]
            if status in ACTIVE_STATES:
                has_active_job = True
                break
            elif status in ["finished", "cancelled", "failed"]:
                exit_code += 1

        if not has_active_job and exit_code == len(jobs):
            empty_poll_count += 1
            logging.info(
                f"No active jobs for run {current_run_id}, poll count: {empty_poll_count}"
            )
            if empty_poll_count >= MAX_EMPTY_POLLS:
                logging.info(
                    f"Run {current_run_id} appears complete, switching back to run polling."
                )
                current_run_id = None
                current_prompt = None
                empty_poll_count = 0
                time.sleep(1)
                continue
            time.sleep(1)
            continue
        else:
            empty_poll_count = 0

        for job in jobs:
            job_id = job["job_id"]
            status = job["status"]
            logging.info(
                f"Job id: {job_id}, status: {status}, remaining jobs: {len(jobs)}"
            )
            if status == "ready":
                device = job.get("device") or {}
                robot_id = device.get("robot_id")
                if not robot_id:
                    logging.warning(
                        f"Job {job_id} is ready but missing robot_id, skipping this poll."
                    )
                    continue
                process_job(
                    client,
                    gpu_client,
                    job_id,
                    robot_id,
                    image_size,
                    image_type,
                    action_type,
                    duration,
                    prompt=current_prompt,
                    task_name=task_name,
                )

        time.sleep(1)


def _state_is_ready(state: dict | None) -> bool:
    return bool(
        state
        and state.get("state") == "normal"
        and state.get("pending_actions") == 0
    )


def run_local_client_loop(
    client,
    policy,
    image_size,
    image_type,
    action_type,
    duration,
    instruction: str = "",
    max_steps: int | None = 50,
    max_wait: int | None = 600,
    idle_sleep_s: float = 0.5,
):
    completed_steps = 0
    start_time = time.time()
    try:
        while True:
            client.start_motion()
            state = client.get_state(image_size, image_type, action_type)
            if not state:
                time.sleep(idle_sleep_s)
                continue
            if state.get("state") == "size_none":
                client.post_size()
                time.sleep(idle_sleep_s)
                continue
            if not _state_is_ready(state):
                if (
                    max_wait is not None
                    and time.time() - start_time > max_wait
                ):
                    break
                time.sleep(idle_sleep_s)
                continue
            if "timestamp" in state:
                logging.info(
                    "get_robot_state time: %.2f",
                    time.time() - state["timestamp"],
                )
            state["job_id"] = f"test_job_id_{start_time}"
            state["task_name"] = "test_job"
            actions = policy.infer(state, prompt=instruction)
            logging.info(f"{completed_steps} Inference result: {len(actions)}")
            client.post_actions(actions, duration, action_type)
            completed_steps += 1
            if max_steps is not None and completed_steps >= max_steps:
                break
            if max_wait is not None and time.time() - start_time > max_wait:
                break
    finally:
        client.end_motion()
