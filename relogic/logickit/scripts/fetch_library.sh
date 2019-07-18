
LIBS_PATH="libs"
if [ ! -d $LIBS_PATH ]; then
  mkdir -p $LIBS_PATH
fi

# Get srl-conll package.
wget -O "${LIBS_PATH}/srlconll-1.1.tgz" http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz
tar xf "${LIBS_PATH}/srlconll-1.1.tgz" -C "./libs"
rm "${LIBS_PATH}/srlconll-1.1.tgz"