
while getopts "m:" arg
do
	case $arg in
		m)  echo "m : $OPTARG"
      sbatch ./submitarc/$OPTARG.sh
      ;;
    ?)
			echo "unknown argument"
	esac
done
