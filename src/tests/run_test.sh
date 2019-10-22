test_script=$1
rm pp_plots/*
make
python $test_script
open pp_plots/*
