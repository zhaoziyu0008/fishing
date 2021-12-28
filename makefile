OBJ = BoostedSieve EnumSieve NVSieve LayerSieve nSieve LDGaussSieve fishing 
CXX = g++
CFLAGS = -O3 -g -funroll-loops -ftree-vectorize -fopenmp -mfma -mavx2 -mavx512f -mavx512vl -march=native -pthread -std=c++11
TOOLCHAINS = -lntl -lgmp -lm -lgf2x
LIBDIR = -I/home/zyzhao/work/packages/include -L/home/zyzhao/work/packages/lib

BoostedSieve: BoostedSieve.cpp
	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o BoostedSieve 

EnumSieve: EnumSieve.cpp
	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o EnumSieve 

NVSieve: NVSieve.cpp
	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o NVSieve

LayerSieve: LayerSieve.cpp
	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o LayerSieve

nSieve: nSieve.cpp
	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o nSieve

LDGaussSieve: LDGaussSieve.cpp
	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o LDGaussSieve

fishing: fishing.cpp
	$(CXX) $(CFLAGS) $< $(LIBDIR) $(TOOLCHAINS) -o fishing

clean:
	-rm $(OBJ)
	
#I have to prepare for the final exam next week....
#after that I will first implement NV-sieve as a baseline, maybe with some features in g6k to get dim for free, then try to accelerate it by nSieve.
#another thing to try is the boost sieve, may be it can gives a larger dim for free thus asymptotically faster

