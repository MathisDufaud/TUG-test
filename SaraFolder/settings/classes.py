import sys


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.logging = True

    def write(self, message):
        self.terminal.write(message)
        if self.logging:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if self.logging:
            self.log.flush()

    def stop_logging(self):
        self.logging = False
        self.log.close()