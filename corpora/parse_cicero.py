import re
with open("de_inventione.txt", "r") as raw:
    with open("de_inventione_parsed.txt", "w") as parsed:
        pattern = re.compile(r"^(\[\s*\d+\s*])")
        idx = 1
        for line in raw.readlines():
            if pattern.match(line):
                print(idx)
                idx += 1

