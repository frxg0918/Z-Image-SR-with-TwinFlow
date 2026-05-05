import torch

dtype = torch.bfloat16

a = torch.tensor(0.8, dtype=dtype)
b = torch.tensor(0.7, dtype=dtype)
print(f"a = {a}, dtype={a.dtype}")
print(f"b = {b}, dtype={b.dtype}")
print("a*b =", a*b, " of dtype", (a*b).dtype)

b = torch.tensor(0.7, dtype=torch.float32)
print(f"a = {a}, dtype={a.dtype}")
print(f"b = {b}, dtype={b.dtype}")
print("a*b =", a*b, " of dtype", (a*b).dtype)

a = torch.tensor(0.8, dtype=torch.float32)
print(f"a = {a}, dtype={a.dtype}")
print(f"b = {b}, dtype={b.dtype}")
print("a*b =", a*b, " of dtype", (a*b).dtype)

a = torch.tensor(0.8, dtype=dtype)
a = a.to(torch.float32)
print(f"a is originally of dtype {dtype}, after to(torch.float32), dtype={a.dtype}")
print("a*b =", a*b, " of dtype", (a*b).dtype)

a = torch.tensor(0.8, dtype=dtype)
b = torch.tensor(0.7, dtype=torch.float32)
c = torch.tensor(0.7, dtype=dtype)
print(f"a = {a}, dtype={a.dtype}")
print(f"b = {b}, dtype={b.dtype}")
print("a + b =", a+b, " of dtype", (a+b).dtype)
print("a + b*c =", a + b*c, " of dtype", (a + b*c).dtype)

for i in range(1001):
    a = torch.tensor(i, dtype=dtype)
    if a-3 == a:
        print(f"Found i={i} where a-2 == a for a of dtype {dtype}, a={a}")
