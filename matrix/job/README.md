matrix job clear # remove all jobs
matrix job stop 
matrix job start
matrix job submit "{'job_id': test, 'task_definitions': [{'func': 'matrix.job.job_utils.echo', 'args': ['hello']}]}"
matrix job status test
matrix job get_results test
