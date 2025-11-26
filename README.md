import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------
# 1) YOUR DATA
# ----------------------------------
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

nps_values = [18500, 19000, 20000, 21500, 21000, 22000]   # the line values

answered = [80, 95, 70, 90, 85, 100]     # number of people who answered the form
total_clients = [200, 220, 210, 250, 240, 260]  # total clients

# X positions for months
x = np.arange(len(months))

plt.figure(figsize=(12, 6))

# ----------------------------------
# 2) BAR WIDTH & POSITIONS
# ----------------------------------
bar_width = 0.25

x_bar1 = x - bar_width/2     # answered
x_bar2 = x + bar_width/2     # total clients

# ----------------------------------
# 3) DRAW THE BARS
# ----------------------------------
plt.bar(x_bar1, answered, width=bar_width, color="yellow", label="Answered")
plt.bar(x_bar2, total_clients, width=bar_width, color="red", label="Total Clients")

# ----------------------------------
# 4) DRAW THE LINE ON TOP
# ----------------------------------
plt.plot(x, nps_values, marker="o", color="black", linewidth=2, label="NPS")

# TEXT LABEL ABOVE EACH POINT
for xi, yi in zip(x, nps_values):
    plt.text(xi, yi, str(yi), ha="center", va="bottom")

# ----------------------------------
# 5) AXES & STYLING
# ----------------------------------
plt.xticks(x, months)

plt.ylim(18000, 23000)   # custom Y scale for line
plt.yticks(range(18000, 23001, 1000))

plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.show()