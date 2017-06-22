OBJS = common.o build_model.o data_IO.o Osiris_train.o Osiris_detect.o Osiris.o
CC = g++
DEBUG = -g
LIBFLAGS = -L Penthus/ -l Penthus -fopenmp
CXXFLAGS = -Wall -c -O3 -std=c++11 -fopenmp $(DEBUG)
LFLAGS = -Wall -O3 $(DEBUG)


MAIN_EXECUTABLE = bin/Osiris

$(MAIN_EXECUTABLE) : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o bin/Osiris $(LIBFLAGS)

common.o : src/common.h src/common.cpp
	$(CC) $(CXXFLAGS) src/common.cpp

data_IO.o : src/data_IO.h src/data_IO.cpp
	$(CC) $(CXXFLAGS) src/data_IO.cpp

build_model.o : src/build_model.h src/build_model.cpp src/data_IO.h
	$(CC) $(CXXFLAGS) src/build_model.cpp $(LIBFLAGS)

Osiris_train.o : src/Osiris_train.h src/Osiris_train.cpp src/data_IO.h src/build_model.h src/common.h
	$(CC) $(CXXFLAGS) src/Osiris_train.cpp $(LIBFLAGS)

Osiris_detect.o : src/Osiris_detect.h src/Osiris_detect.cpp src/build_model.h src/common.h
	$(CC) $(CXXFLAGS) src/Osiris_detect.cpp $(LIBFLAGS)

Osiris.o : src/Osiris.cpp src/Osiris_train.h src/data_IO.h src/build_model.h
	$(CC) $(CXXFLAGS) src/Osiris.cpp $(LIBFLAGS)

clean:
	rm $(OBJS) $(MAIN_EXECUTABLE)