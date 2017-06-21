
base = {
    0:"nul",
    1:"een",
    2:"twee",
    3:"drie",
    4: "vier",
    5: "vijf",
    6: "zes",
    7: "zeven",
    8: "acht",
    9: "negen",
    10: "tien",
    11: "elf",
    12: "twaalf",
    13: "dertien",
    14: "veertien",
    15: "vijftien",
    16: "zestien",
    17: "zeventien",
    18: "achtien",
    19: "negentien",
}

tens = {
    2: "twintig",
    3: "dertig",
    4: "veertig",
    5: "vijftig",
    6: "zestig",
    7: "zeventig",
    8: "tachtig",
    9: "negentig",
}

pow = base.copy()
del pow[0]
pow[1] = ""

def dutch_num_to_text(num):
    numstr = str(num)
    l = len(numstr)
    if l > 1: numstr = numstr.lstrip("0")
    l = len(numstr)
    num = int(numstr)
    if num < 20: return base[num]

    if l == 2:
        b = int(numstr[1])
        t = int(numstr[0])
        if b == 0: return tens[t]
        return base[b] + " " + tens[t]
    elif l == 3:
        h = int(numstr[0])
        return pow[h]+" honderd "+dutch_num_to_text(int(numstr[1:]))
    elif l == 4:
        d = int(numstr[0])
        return pow[d] + " duizend " + dutch_num_to_text(int(numstr[1:]))
    elif l == 5:
        return dutch_num_to_text(numstr[:2]) + " duizend " + dutch_num_to_text(int(numstr[1:]))
    elif l == 6:
        return dutch_num_to_text(numstr[:3]) + " duizend " + dutch_num_to_text(int(numstr[2:]))

    return str(num)


if __name__ == '__main__':
    import random
    for i in range(200):
        print i, dutch_num_to_text(i)

    for i in range(100):
        r = long(random.random()* 1000000)
        print r, dutch_num_to_text(r)

