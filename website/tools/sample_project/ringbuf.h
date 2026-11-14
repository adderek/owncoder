#ifndef RINGBUF_H
#define RINGBUF_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *data;
    size_t   cap;
    size_t   head;   /* next write */
    size_t   tail;   /* next read  */
    size_t   count;
} ringbuf_t;

int    ringbuf_init(ringbuf_t *rb, size_t cap);
void   ringbuf_free(ringbuf_t *rb);
size_t ringbuf_write(ringbuf_t *rb, const uint8_t *src, size_t n);
size_t ringbuf_read(ringbuf_t *rb, uint8_t *dst, size_t n);

#endif
