
EXEC = ojectdetector
INCLUDE = -Idarknet/include
INCLUDE+= -I/opt/mvIMPACT_acquire/
FLAGS = -Wno-write-strings
FLAGS+= -O2 -Wall -W -fPIC -D_REENTRANT -D_GNU_SOURCE  -DNDEBUG -fvisibility=hidden
LIB= darknet/libdarknet.a -lm
LIB+= `pkg-config --libs opencv`
LIB+= -L/opt/mvIMPACT_acquire/apps/../lib/x86_64 -lmvDeviceManager -lmvPropHandling
COMMON=-DOPENCV

all:
		g++ -c objectdetector.cpp $(COMMON) $(FLAGS) $(INCLUDE)
		g++ -o $(EXEC) objectdetector.o $(LIB) $(COMMON)

clean:
		rm *.o
		rm $(EXEC)
