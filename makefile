OBJ = tool.o basis.o UidHashTable.o pool.o NVSieve.o 3Sieve.o BoostedSieve.o
CXX = g++
CFLAGS = -O3 -g -funroll-loops -ftree-vectorize -fopenmp -mfma -mavx2 -mavx512f -mavx512vl -march=native -pthread -std=c++11
TOOLCHAINS = -lntl -lgmp -lm -lgf2x
LIBDIR = -I/home/zyzhao/work/packages/include -L/home/zyzhao/work/packages/lib

all: test

test: test.cpp $(OBJ)
	$(CXX) $(CFLAGS) $^ $(LIBDIR) $(TOOLCHAINS) -o test

pool.o: pool.cpp 
	$(CXX) $(CFLAGS) $^ $(LIBDIR) $(TOOLCHAINS) -c

UidHashTable.o: UidHashTable.cpp 
	$(CXX) $(CFLAGS) $^ $(LIBDIR) $(TOOLCHAINS) -c

basis.o: basis.cpp 	
	$(CXX) $(CFLAGS) $^ $(LIBDIR) $(TOOLCHAINS) -c

tool.o: tool.cpp	
	$(CXX) $(CFLAGS) $^ $(LIBDIR) $(TOOLCHAINS) -c

NVSieve.o: NVSieve.cpp
	$(CXX) $(CFLAGS) $^ $(LIBDIR) $(TOOLCHAINS) -c

3Sieve.o: 3Sieve.cpp
	$(CXX) $(CFLAGS) $^ $(LIBDIR) $(TOOLCHAINS) -c

#BoostedSieve.o: BoostedSieve.cpp
#	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -c

#nSieve: nSieve.cpp
#	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o nSieve

#EnumSieve: EnumSieve.cpp
#	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o EnumSieve 

#LayerSieve: LayerSieve.cpp
#	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o LayerSieve

#fishing: fishing.cpp
#	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o fishing


clean:
	-rm $(OBJ)

