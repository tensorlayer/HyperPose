TAG = openpose-builder:snapshot


builder:
	docker build --rm -t $(TAG) -f Dockerfile.builder ..

run: builder
	docker run --rm -it $(TAG) bash
