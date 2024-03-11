
#region ANSI colors with end reset
def ANSI_RESET(text):
    return f"\u001B[0m{text}\u001B[0m"


def ANSI_BLACK(text):
    return f"\u001B[30m{text}\u001B[0m"


def ANSI_RED(text):
    return f"\u001B[31m{text}\u001B[0m"


def ANSI_GREEN(text):
    return f"\u001B[32m{text}\u001B[0m"


def ANSI_YELLOW(text):
    return f"\u001B[33m{text}\u001B[0m"


def ANSI_BLUE(text):
    return f"\u001B[34m{text}\u001B[0m"


def ANSI_PURPLE(text):
    return f"\u001B[35m{text}\u001B[0m"


def ANSI_CYAN(text):
    return f"\u001B[36m{text}\u001B[0m"


def ANSI_WHITE(text):
    return f"\u001B[37m{text}\u001B[0m"
#endregion

ansi_rainbow_global_variable = 0

def ANSI_RAINBOW_LINE(text):
    global ansi_rainbow_global_variable
    rainbow_colors = [ANSI_RED, ANSI_YELLOW, ANSI_GREEN, ANSI_BLUE, ANSI_PURPLE]
    if ansi_rainbow_global_variable % len(rainbow_colors) == 0:
        ansi_rainbow_global_variable = 0
    ansi_rainbow_global_variable += 1
    return rainbow_colors[ansi_rainbow_global_variable - 1](text)


