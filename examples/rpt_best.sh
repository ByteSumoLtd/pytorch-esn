cat runlog.csv | head -1 | sed 's/,/\t/g' > tmp ; cat runlog.csv | grep -v train | sort -g -t, -k3,3  | head -5 | sed 's/,/\t/g' >>tmp ; cat tmp | column -t; rm tmp
