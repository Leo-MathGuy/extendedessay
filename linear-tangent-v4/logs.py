import logging


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
        "ADMIN": "\033[45m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname:8}{self.RESET}"
        return super().format(record)


l = logging.getLogger("l")


l.setLevel(logging.DEBUG)

console_formatter = ColorFormatter(
    "\033[0;30m({process})\033[0m \033[1;30m{asctime}\033[0m {levelname:8} {message}",
    style="{",
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_formatter)
l.addHandler(console_handler)
