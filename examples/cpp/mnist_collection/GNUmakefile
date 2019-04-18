all: train_lenet_classifier.cpp train_resnet_classifier.cpp train_vae.cpp train_dcgan.cpp train_siamese.cpp train_vat.cpp
	$(CXX) -std=c++11 -O -o train_lenet_classifier train_lenet_classifier.cpp -lnnabla -lz
	$(CXX) -std=c++11 -O -o train_resnet_classifier train_resnet_classifier.cpp -lnnabla -lz
	$(CXX) -std=c++11 -O -o train_vae train_vae.cpp -lnnabla -lz
	$(CXX) -std=c++11 -O -o train_dcgan train_dcgan.cpp -lnnabla -lz
	$(CXX) -std=c++11 -O -o train_siamese train_siamese.cpp -lnnabla -lz
	$(CXX) -std=c++11 -O -o train_vat train_vat.cpp -lnnabla -lz

lenet: train_lenet_classifier.cpp
	$(CXX) -std=c++11 -O -o train_lenet_classifier train_lenet_classifier.cpp -lnnabla -lz

resnet: train_resnet_classifier.cpp
	$(CXX) -std=c++11 -O -o train_resnet_classifier train_resnet_classifier.cpp -lnnabla -lz

vae: train_vae.cpp
	$(CXX) -std=c++11 -O -o train_vae train_vae.cpp -lnnabla -lz

dcgan: train_dcgan.cpp
	$(CXX) -std=c++11 -O -o train_dcgan train_dcgan.cpp -lnnabla -lz

siamese: train_siamese.cpp
	$(CXX) -std=c++11 -O -o train_siamese train_siamese.cpp -lnnabla -lz

vat: train_vat.cpp
	$(CXX) -std=c++11 -O -o train_vat train_vat.cpp -lnnabla -lz

clean:
	rm -f train_lenet_classifier
	rm -f train_resnet_classifier
	rm -f train_vae
	rm -f train_dcgan
	rm -f train_siamese
	rm -f train_vat
