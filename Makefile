
EXEC = objectdetector
INCLUDE = #-Idarknet/include
INCLUDE+= -I/opt/mvIMPACT_Acquire/
FLAGS = -Wno-write-strings
FLAGS+= -O2 -Wall -W -fPIC -D_REENTRANT -D_GNU_SOURCE  -DNDEBUG -fvisibility=hidden
LIB= darknet/libdarknet.a -lm
LIB+= `pkg-config --libs opencv`
LIB+= -L/opt/mvIMPACT_Acquire/apps/../lib/arm64 -lmvDeviceManager -lmvPropHandling
COMMON=-DOPENCV

# GPU
COMMON+= -DGPU -I/usr/local/cuda/include/
LIB+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand

COMMON+= -DCUDNN
LIB+= -lcudnn
LIB+= -lstdc++
LIB+= -shared-libgcc

all:
		g++ -c objectdetector.cpp $(COMMON) $(FLAGS) $(INCLUDE)
		g++ -o $(EXEC) objectdetector.o $(LIB) $(COMMON)

clean:
		rm *.o
		rm $(EXEC)
