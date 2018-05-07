set -e

detdir=$1
if [ -z "$detdir" ];
then
   detdir=DALY_detectron_allAnn
   #detdir=UCF101_detectron_allAnn
fi

missfromlist=$2
#missfromlist=/sequoia/data2/gcheron/UCF101/detection/OF_vidlist_all.txt

CHECK_MISSING=1

while true
do
   rm -rf .finish_list_tmp
   for vid in /sequoia/data2/jalayrac/nips2017weakpose/$detdir/*
   do
      if $(echo $vid | grep -q DALY)
      then
         imdir=/sequoia/data2/gcheron/DALY/images
      elif $(echo $vid | grep -q UCF101)
      then
         imdir=/sequoia/data2/gcheron/UCF101/images 
      else
         ERROR
      fi
   
      vidname=$(basename $vid)
   
      numImages=$(find $imdir/$vidname/ -name "*.jpg" | wc -l)
      numDet=$(find $vid/ -name "*.mat" | wc -l)
   
      if [[ "$numImages" -eq "$numDet" ]]
      then
         echo $vid >> .finish_list_tmp
      elif [[ "$CHECK_MISSING" -eq "1" ]]
      then
         echo "$vid is missing"
      fi
      if [[ "$CHECK_MISSING" -ne "1" ]]
      then
         echo $imdir $vidname $numImages $numDet
      fi
   done
   mv .finish_list_tmp finish_list_${detdir}.txt

   if [ ! -z "$missfromlist" ]
   then
      while read line
      do
         vname=$(echo $line | cut -f1 -d " ")
         grep -q "$vname" finish_list_${detdir}.txt
         NOTFOUND=$?
         if [ $NOTFOUND -eq "1" ]
         then
            echo $vname
         fi

      done < "$missfromlist"
   fi

   if [[ "$CHECK_MISSING" -eq "1" ]]
   then
      exit 0
   fi

   sleep 600
done
