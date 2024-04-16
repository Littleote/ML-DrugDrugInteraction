#!/usr/bin/env bash
#
# Runs Stanford CoreNLP server

# set this path to the directory where you decompressed StanfordCore
STANFORDDIR=$(cat run.config | head -3 | tail -1)

if [ -f /tmp/corenlp-server.running ]; then
    echo "server already running"
else
    echo java -mx5g -cp \"$STANFORDDIR/*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer $*
    java -mx5g -cp "$STANFORDDIR/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer $* &
    echo $! > ./tmp/corenlp-server.running
    wait
    rm ./tmp/corenlp-server.running
fi
