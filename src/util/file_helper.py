def write(path, content):
    with open(path, "a+") as dst_file:
        dst_file.write(content + '\n')


def read2mem(path):
    with open(path) as f:
        content = ''
        while 1:
            try:
                lines = f.readlines(100)
            except UnicodeDecodeError:
                f.close()
                continue
            if not lines:
                break
            for line in lines:
                content += line
    return content


def read_lines(path):
    with open(path) as f:
        content = list()
        while 1:
            try:
                lines = f.readlines(100)
            except UnicodeDecodeError:
                f.close()
                continue
            if not lines:
                break
            for line in lines:
                content.append(line)
    return content
