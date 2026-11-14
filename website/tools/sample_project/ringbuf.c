#include <stdlib.h>
#include "ringbuf.h"

int ringbuf_init(ringbuf_t *rb, size_t cap)
{
    rb->data = malloc(cap);
    if (!rb->data)
        return -1;
    rb->cap = cap;
    rb->head = rb->tail = rb->count = 0;
    return 0;
}

void ringbuf_free(ringbuf_t *rb)
{
    free(rb->data);
    rb->data = NULL;
}

size_t ringbuf_write(ringbuf_t *rb, const uint8_t *src, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) {
        rb->data[rb->head] = src[i];
        rb->head = (rb->head + 1) % rb->cap;
        rb->count++;
    }
    return i;
}

size_t ringbuf_read(ringbuf_t *rb, uint8_t *dst, size_t n)
{
    size_t i;
    for (i = 0; i < n && rb->count > 0; i++) {
        dst[i] = rb->data[rb->tail];
        rb->tail = (rb->tail + 1) % rb->cap;
        rb->count--;
    }
    return i;
}
