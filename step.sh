#! /bin/bash

BASEDIR=$(cat run.config | head -1 | tail -1)
PYTHON=$(cat run.config | head -2 | tail -1)

if [ "$1" == "regen" ]
then
    ./corenlp-server.sh -quiet true -port 9000 -timeout 15000  &
    sleep 1

    # extract features
    echo "Extracting features"
    rm -rf features
    mkdir -p features
    "$PYTHON" extract_features.py $BASEDIR/data/devel/ features/devel_{}.cod.cl > devel.cod
    "$PYTHON" extract_features.py $BASEDIR/data/train/ features/train_{}.cod.cl > train.cod

    kill `cat ./tmp/corenlp-server.running`
fi

# Step features
echo "Stepping model features"
"$PYTHON" step.py features\\train_{}.cod.cl train.cod features\\devel_{}.cod.cl devel.cod $BASEDIR/data/devel/ step.csv
