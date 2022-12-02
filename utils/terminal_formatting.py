
class TerminalFormatting:
    # Recommended usage:
    #   from terminal_formatting import TerminalFormatting as tf
    # More info: https://ss64.com/nt/syntax-ansi.html

    FG_BLACK        = "\033[30m"
    FG_RED          = "\033[31m"
    FG_GREEN        = "\033[32m"
    FG_YELLOW       = "\033[33m"
    FG_BLUE         = "\033[34m"
    FG_PURPLE       = "\033[35m"
    FG_DARK_CYAN    = "\033[36m"
    FG_GRAY         = "\033[37m"
    FG_DARK_GRAY    = "\033[90m"
    FG_PINK         = "\033[91m"
    FG_LIGHT_GREEN  = "\033[92m"
    FG_LIGHT_YELLOW = "\033[93m"
    FG_LIGHT_BLUE   = "\033[94m"
    FG_MAGENTA      = "\033[95m"
    FG_CYAN         = "\033[96m"
    FG_WHITE        = "\033[97m"

    BG_BLACK        = "\033[40m"
    BG_RED          = "\033[41m"
    BG_GREEN        = "\033[42m"
    BG_YELLOW       = "\033[43m"
    BG_BLUE         = "\033[44m"
    BG_PURPLE       = "\033[45m"
    BG_DARK_CYAN    = "\033[46m"
    BG_GRAY         = "\033[47m"
    BG_DARK_GRAY    = "\033[100m"
    BG_PINK         = "\033[101m"
    BG_LIGHT_GREEN  = "\033[102m"
    BG_LIGHT_YELLOW = "\033[103m"
    BG_LIGHT_BLUE   = "\033[104m"
    BG_MAGENTA      = "\033[105m"
    BG_CYAN         = "\033[106m"
    BG_WHITE        = "\033[107m"

    BOLD         = "\033[1m"
    UNDERLINE    = "\033[4m"
    NO_UNDERLINE = "\033[24m"
    REVERSE      = "\033[7m"
    NO_REVERSE   = "\033[27m"

    DEFAULT      = "\033[0m"
