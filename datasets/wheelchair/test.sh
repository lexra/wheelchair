#!/bin/bash -e

N=0
for F in `ls |grep '\.txt' | awk -F '.txt' '{print $1}'` ; do
	T=`printf "%05d\n" $N`

	mv -- "${F}.txt" "${T}.txt"
	mv -- "${F}.jpg" "${T}.jpg"

	N=$(($N + 1))
done
