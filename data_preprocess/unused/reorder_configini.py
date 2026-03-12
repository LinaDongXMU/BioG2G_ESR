import configparser
import sys


def read(p):
    with open(p, 'r') as f:
        data = f.readlines()
    result = {}
    for line in data:
        line = line.strip()
        if line.startswith('['):
            section_name = line[1:-1]
            if section_name not in result:
                result[section_name] = {}
        elif line:
            key, value = line.split('=')
            result[section_name][key.strip()] = value.strip()
    return result['DEFAULT']

def write(d, p):
    config = configparser.ConfigParser()
    config['DEFAULT'] = d
    with open(p, 'w') as f:
        config.write(f)


if __name__ == '__main__':
    p1 = sys.argv[1]
    p2 = sys.argv[2]
    p = sys.argv[3]
    d1 = read(p1)
    d2 = read(p2)
    new_dict = {}
    for k in d2.keys():
        if k in d1.keys():
            new_dict[k] = d1[k]
    for k in d1.keys():
        if k not in d2.keys():
            new_dict[k] = d1[k]
    write(new_dict, p)