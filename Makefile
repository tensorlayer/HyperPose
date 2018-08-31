pafprocess:
	make -C inference/pafprocess

cpp_examples:
	make -C cpp


all: \
	pafprocess \
	cpp_examples
