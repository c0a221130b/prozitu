import re
with open("./num2/num2", mode="rb") as f:
    b = f.read()
    res = re.search(b"\x48\x83\xc4\x08", b)
if res:
    print("found.")
else:
    print("not found.")