# ringbuf

Small fixed-capacity byte ring buffer in C.

- `ringbuf_init` / `ringbuf_free` manage storage
- `ringbuf_write` appends bytes
- `ringbuf_read` consumes bytes in FIFO order

Build with `make`; links as `libringbuf.a`.
