all:dependency

dependency:
	pip3 install -r requirement.txt
	sudo apt install libxerces-c3.2 libjpeg8
	$(MAKE) -C PCDet

clean:
	rm -rf raw_data/ log/