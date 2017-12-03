#!/usr/bin/env bash
#
# Runs the English PCFG parser on one or more files, printing dependency parses only

if [ ! $# -ge 1 ]; then
  echo Usage: `basename $0` 'file(s)'
  echo
  exit
fi

scriptdir=`dirname $0`

java -Xmx8g -cp "$scriptdir/stanford-parser-full-2017-06-09/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -nthreads 1 -sentences newline \
 -retainTmpSubcategories -outputFormat "typedDependencies" -outputFormatOptions "basicDependencies" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $*
