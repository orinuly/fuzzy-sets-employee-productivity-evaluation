import numpy as np
import matplotlib.pyplot as plt
import membership as mem

# Fuzzification
tct = np.linspace(0, 60, 200)
short = np.array([mem.sigmoid(x, -0.5, 15) for x in tct])
moderate = np.array([mem.gauss(x, 25, 4) for x in tct])
long = np.array([mem.sigmoid(x, 0.6, 30) for x in tct])

wes = np.linspace(0, 100, 200)
unpleasant = np.array([mem.sigmoid(x, -0.5, 30) for x in wes])
neutral = np.array([mem.gauss(x, 45, 4) for x in wes])
pleasant = np.array([mem.sigmoid(x, 0.6, 60) for x in wes])

coffee = np.linspace(0, 5, 200)
low = np.array([mem.sigmoid(x, -5, 1) for x in coffee])
middle = np.array([mem.gauss(x, 1.75, 0.4) for x in coffee])
high = np.array([mem.sigmoid(x, 6, 2) for x in coffee])

productivity = np.linspace(0, 100, 200)
low_pr = np.array([mem.sigmoid(x, -0.5, 30) for x in productivity])
middle_pr = np.array([mem.gauss(x, 45, 4) for x in productivity])
high_pr = np.array([mem.sigmoid(x, 0.6, 60) for x in productivity])

inp_tct = 5
inp_wes = 50
inp_cf = 2

# Modeling Input
inp_tct_sh = mem.sigmoid(inp_tct, -0.5, 15)
inp_tct_m = mem.gauss(inp_tct, 25, 4)
inp_tct_l = mem.sigmoid(inp_tct, 0.6, 30)

inp_wes_unpl = mem.sigmoid(inp_wes, -0.5, 30)
inp_wes_neut = mem.gauss(inp_wes, 45, 4)
inp_wes_pl = mem.sigmoid(inp_wes, 0.6, 60)

inp_cf_l = mem.sigmoid(inp_cf, -5, 1)
inp_cf_m = mem.gauss(inp_cf, 1.75, 0.4)
inp_cf_h = mem.sigmoid(inp_cf, 6, 2)

# RULE 1
# IF Task Completion Time is Long AND Work Environment Satisfaction is Unpleasant THEN Productivity Level is Low.
ante = np.min([inp_tct_l, inp_wes_unpl])
r1 = np.fmin(ante, low_pr)

# RULE 2
# IF Caffeine Intake is High THEN Productivity Level is Moderate.
ante = np.min([inp_cf_h])
r2 = np.fmin(ante, middle_pr)

# RULE 3
# IF Task Completion Time is Short AND Work Environment Satisfaction is Pleasant THEN Productivity Level is High.
ante = np.min([inp_tct_sh, inp_wes_pl])
r3 = np.fmin(ante, high_pr)

# RULE 4
# IF Work Environment Satisfaction is Neutral AND Caffeine Intake is Moderate THEN Productivity Level is Moderate.
ante = np.min([inp_wes_neut, inp_cf_m])
r4 = np.fmin(ante, middle_pr)

# RULE 5
# IF Caffeine Intake is Low THEN Productivity Level is Low.
ante = np.min([inp_cf_l])
r5 = np.fmin(ante, low_pr)

# RULE 6
# IF Task Completion Time is Moderate AND Work Environment Satisfaction is Pleasant THEN Productivity Level is High.
ante = np.min([inp_tct_m, inp_wes_pl])
r6 = np.fmin(ante, high_pr)

# RULE 7
# IF Task Completion Time is Short THEN Productivity Level is High.
ante = np.min([inp_tct_sh])
r7 = np.fmin(ante, high_pr)

# Aggregate all the rules
aggregated = np.maximum(r1, np.maximum(r2, np.maximum(r3, np.maximum(r4, np.maximum(r5, np.maximum(r6, r7))))))

# Defuzzification using the Centroid method
output_productivity = np.trapz(aggregated * productivity, productivity) / np.trapz(aggregated, productivity)

print("Employee Productivity Level =", output_productivity)

plt.figure(figsize=(12, 8))

# Task Completion Time
plt.subplot(3, 2, 1)
plt.plot(tct, short, label='Short')
plt.plot(tct, moderate, label='Moderate')
plt.plot(tct, long, label='Long')
plt.scatter(inp_tct, 0)
plt.title('Task Completion Time')
plt.legend()

# Work Environment Satisfaction
plt.subplot(3, 2, 2)
plt.plot(wes, unpleasant, label='Unpleasant')
plt.plot(wes, neutral, label='Neutral')
plt.plot(wes, pleasant, label='Pleasant')
plt.scatter(inp_wes, 0)
plt.title('Work Environment Satisfaction')
plt.legend()

# Caffeine Intake
plt.subplot(3, 2, 3)
plt.plot(coffee, low, label='Low')
plt.plot(coffee, middle, label='Middle')
plt.plot(coffee, high, label='High')
plt.scatter(inp_cf, 0)
plt.title('Caffeine Intake')
plt.legend()

# Productivity
plt.subplot(3, 2, 4)
plt.plot(productivity, low_pr, label='Low')
plt.plot(productivity, middle_pr, label='Middle')
plt.plot(productivity, high_pr, label='High')
plt.scatter(output_productivity, 0)
plt.title('Productivity')
plt.legend()

# Plots
plt.tight_layout()
plt.show()
