import re


def remove_format(text: str) -> str:
    # if the accout of \t, \r, \v, \f is less than 3, replace \t, \r, \v, \f with space
    if len(re.findall(r"\s", text.replace(" ", ""))) > 3:
        # in case this is a line of a table
        return text
    return re.sub(r"\s", " ", text)


# remove newlines
def get_cleaned_paragraph(s: str) -> str:
    text = str(s)
    text = re.sub(r"\n{3,}", r"\n", text)  # replace \n\n\n... with \n
    text = re.sub("\n\n", "", text)
    lines = text.split("\n")
    lines_remove_format = [remove_format(line) for line in lines]
    return lines_remove_format
