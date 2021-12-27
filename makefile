OBJ = BoostedSieve EnumSieve NVSieve LayerSieve SubSieve LDGaussSieve fishing 
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

