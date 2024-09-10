def escape_markdown(text: str) -> str:
    # Escapes markdown special characters: underscore, asterisk, backtick, square brackets
    escape_chars = '_*`['
    return ''.join(['\\' + char if char in escape_chars else char for char in text])
