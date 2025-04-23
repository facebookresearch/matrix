# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
matrix job to launch many related tasks.

A python package for user to submit a list of tasks to be run by Ray.
Each task has resource requirements and a user defined function that return (bool, json data). 
Run some num of tasks concurrently based on Ray resources available. 
Users can use job_id to query the progress. If a task failed, it can be retried up to 3 times. 
Job results format is {task_id: (bool, json_data)}.
1. each job submission have a max concurrency to incrementally schedule at most k tasks together.
2. add a timeout in the task definition, once it become running, start count the time it took and force kill the job it is excced timeout.
3. each task need a list of application to deploy by calling an deploy_applications api. the deploy can take time from 5 minute to an hour. 
   there is another api to check if the deploy succeded check_health. only when all applications are healthy, 
   proceed to run the user function. The third function will cleanup the application once task is done.
4. Create a run method to monitor the progress of task status in a thread. 
   Users can submit and check status for job, but the manager handles job in serial, 
   so there is a current_job. Checkpoint the state to ray store in case we restart the manager. 
   Use a lock to avoid race, don't hold the lock too long, which prevent submit jobs and check status.

fault tolerance:
1. deployed app in task or job shoul be undeployed after task/job done or killed.
"""
