
while getopts "m:" arg
do
	case $arg in
		m)  echo "m : $OPTARG"
      sbatch ./submitjade/$OPTARG.sh
      ;;
    ?)
			echo "unknown argument"
	esac
done
