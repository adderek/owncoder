class Resource:
    def __init__(self):
        self.open = False

    def acquire(self):
        self.open = True

    def release(self):
        self.open = False

    # missing: __enter__ / __exit__ so this can be used in a `with` block
