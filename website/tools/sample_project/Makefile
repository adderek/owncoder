CC ?= cc
CFLAGS ?= -Wall -Wextra -O2

libringbuf.a: ringbuf.o
	$(AR) rcs $@ $^

ringbuf.o: ringbuf.c ringbuf.h

clean:
	rm -f *.o *.a
