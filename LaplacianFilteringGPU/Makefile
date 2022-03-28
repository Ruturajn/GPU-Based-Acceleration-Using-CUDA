CC = g++
STD = -std
CSTD = c++11
FILE = one
CPPFLAGS = `pkg-config --cflags --libs opencv`
SRC = Vid_Edge.cpp

.PHONY: all

all : $(FILE).out

$(FILE).out :
	$(CC) $(STD)=$(CSTD) $(SRC) $(CPPFLAGS) -o $(FILE).out

clean:
	rm -f *.out
