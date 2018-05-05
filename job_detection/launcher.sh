#!/bin/bash 

QSTAT="qstat"
QSUB="qsub"
USER=gcheron 

jobnbpath="nb_jobs.txt"
job_list="job_list"
sub_list="submitted_jobs"

update_job_number ()
{
        cur_qstat=$($QSTAT)
        if (( $? ))
        then
           echo "gdi error detected" # gid error happened 
           return 1
        fi

        COUNTERTOTAL=`echo "$cur_qstat" | grep $USER | wc -l` # number of jobs including qlogin --> this grep for CPU/GPU for the moment (note that if you just add a "| grep gpu" you will not count the jobs in queue: qw state)
        COUNTERQSUB=`echo "$cur_qstat" | grep $USER | grep -v QLOGIN | wc -l` # number of jobs using qsub

	JOBID=$(echo $JOBLIST | cut -f 1 -d " ") # take the first (older) job

        info=$(cat $jobnbpath | awk '{print $1}')
        qsub_num_min=$(echo $info | cut -f 1 -d " ")
        q_num_max=$(echo $info | cut -f 2 -d " ")
}


err1 ()
{
        echo "error found in 'update_job_number'... try to restart in 20 s"
        sleep 20
        update_job_number || err1
}

while true; do
	JOBLIST=$(ls -rt $job_list/*.pbs 2> /dev/null) # get the job list from the folder
	
	if [ -z $JOBLIST 2> /dev/null ]
	#if [ -z $JOBLIST ]
	then
		echo "Empty job list!"
		exit 0
	else 
		update_job_number || err1
		echo "$COUNTERTOTAL >= $q_num_max and $COUNTERQSUB >= $qsub_num_min"
		if (( "$COUNTERTOTAL" >= "$q_num_max" && "$COUNTERQSUB" >= "$qsub_num_min" ));
		then 
			echo "maximum job number reached - total: $COUNTERTOTAL/$q_num_max - qsub: $COUNTERQSUB/$qsub_num_min"
                        echo "pending job: $JOBID"
                        date +"%d-%m [%T]"

			while (( "$COUNTERTOTAL" >= "$q_num_max" && "$COUNTERQSUB" >= "$qsub_num_min" )); do 
				sleep 20 
				update_job_number || err1
			done
			echo "Restart to launch jobs..." 
		fi 
	
		echo "submitting: $JOBID"
		$QSUB $JOBID 
                mv $JOBID "$sub_list/"
		sleep 1
	fi
done
