import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("playtime_purchase_effect/part-00000-*.csv")

order = ["<2h", "2-10h", "10-50h", "50h+"]
df["playtime_bucket"] = pd.Categorical(df["playtime_bucket"], order)
df = df.sort_values("playtime_bucket")

plt.figure(figsize=(6,4))
plt.plot(df["playtime_bucket"], df["positive_rate"], marker="o")
plt.title("Positive review rate by playtime")
plt.xlabel("Playtime bucket")
plt.ylabel("Positive review rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("positive_rate_by_playtime.png", dpi=200)

plt.figure(figsize=(6,4))
plt.plot(df["playtime_bucket"], df["purchase_rate"], marker="o")
plt.title("Purchase rate by playtime")
plt.xlabel("Playtime bucket")
plt.ylabel("Purchase rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("purchase_rate_by_playtime.png", dpi=200)

print("Plots saved.")
