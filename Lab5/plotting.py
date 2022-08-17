import matplotlib.pyplot as plt
import csv

epoch = []
kl_weight = []
tfr = []
loss = []
psnr = []

# Epoch	Accuracy	Loss_D	tLoss_G	D(x)	D(G(z))

  
with open('epoch_curve_plotting_data.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')

    for row in lines:
        epoch.append(int(row[0]))
        kl_weight.append(float(row[1]))
        tfr.append(float(row[2]))
        loss.append(float(row[3]))
        psnr.append(float(row[4]))

fig = plt.figure()

plt.subplot(2, 2, 1)
plt.plot(epoch, kl_weight, color = 'g', linestyle = 'dashed', label = "KL Weight")
plt.xlabel("Epoch")
plt.ylabel("KL Weight")
plt.title("KL Weight")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epoch, tfr, color = 'b', linestyle = 'dashed', label = "TFR")
plt.xlabel("Epoch")
plt.ylabel("TFR")
plt.title("Teacher Forcing Ratio")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epoch, loss, color = 'y', linestyle = 'dashed', label = "Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(epoch, psnr, color = 'r', label = "PSNR")
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.title("PSNR")
plt.legend()

plt.show()

