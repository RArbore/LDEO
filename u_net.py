import torch
from torchvision import transforms
import random
import time
import math
import os
import sys

manualSeed = int(torch.rand(1).item() * 1000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


DATA_SIZE = 1000 * 1000

DATA_DIMENSIONS = [1000, 1000]

TRAIN_DATA_SIZE = 26

NUM_BATCHES = 13

BATCH_SIZE = int(TRAIN_DATA_SIZE / NUM_BATCHES)

VALIDATION_DATA_SIZE = 6

VALIDATION_NUM_BATCHES = 1

VALIDATION_BATCH_SIZE = int(VALIDATION_DATA_SIZE / VALIDATION_NUM_BATCHES)

TESTING_DATA_SIZE = 6

TESTING_NUM_BATCHES = 1

TESTING_BATCH_SIZE = int(TESTING_DATA_SIZE / TESTING_NUM_BATCHES)

NUM_EPOCHS = 2000

BCE_COEFFICIENT = 10

nf = 16

kernel = 3

padding = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu")

folder = ""


class UNet(torch.nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.s1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, nf, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf, nf, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s2 = torch.nn.Sequential(
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(nf, nf * 2, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf * 2, nf * 2, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s3 = torch.nn.Sequential(
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(nf * 2, nf * 4, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf * 4, nf * 4, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s4 = torch.nn.Sequential(
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(nf * 4, nf * 8, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf * 8, nf * 8, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s5 = torch.nn.Sequential(
            torch.nn.AvgPool2d(5),
            torch.nn.Conv2d(nf * 8, nf * 16, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf * 16, nf * 16, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s6 = torch.nn.Sequential(
            torch.nn.AvgPool2d(5),
            torch.nn.Conv2d(nf * 16, nf * 32, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf * 32, nf * 32, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s7 = torch.nn.Sequential(
            torch.nn.Conv2d(nf * 32, nf * 16, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf * 16, nf * 16, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s8 = torch.nn.Sequential(
            torch.nn.Conv2d(nf * 16, nf * 8, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf * 8, nf * 8, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s9 = torch.nn.Sequential(
            torch.nn.Conv2d(nf * 8, nf * 4, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf * 4, nf * 4, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s10 = torch.nn.Sequential(
            torch.nn.Conv2d(nf * 4, nf * 2, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(nf * 2, nf * 2, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s11 = torch.nn.Sequential(
            torch.nn.Conv2d(nf * 2, 2, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(2, 2, kernel, 1, padding),
            torch.nn.Softmax(dim=1),
        )
        self.upconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nf * 32, nf * 16, 5, 5)
        )
        self.upconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nf * 16, nf * 8, 5, 5)
        )
        self.upconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nf * 8, nf * 4, 2, 2)
        )
        self.upconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nf * 4, nf * 2, 2, 2)
        )
        self.upconv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nf * 2, nf, 2, 2)
        )

    def forward(self, input):
        before = input.view(input.size(0), 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1])

        s1 = self.s1(before)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        s5 = self.s5(s4)
        s6 = self.s6(s5)
        s7 = self.s7(torch.cat((s5, self.upconv1(s6)), dim=1))
        s8 = self.s8(torch.cat((s4, self.upconv2(s7)), dim=1))
        s9 = self.s9(torch.cat((s3, self.upconv3(s8)), dim=1))
        s10 = self.s10(torch.cat((s2, self.upconv4(s9)), dim=1))
        s11 = self.s11(torch.cat((s1, self.upconv5(s10)), dim=1))

        out = s11[:, 0, :, :].view(input.size(0), 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1])
        return out * torch.tensor(0.9998) + torch.tensor(0.0001)


def save_image(tensor, filename):
    ndarr = tensor.mul(255).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)


def pixel_BCE(output_tensor, label_tensor):
    label_tensor = torch.min(label_tensor, torch.ones(label_tensor.size()).float().to(device))

    loss_tensor = BCE_COEFFICIENT * label_tensor * torch.log(output_tensor) + (torch.ones(label_tensor.size()).to(device) - label_tensor) * torch.log(torch.ones(output_tensor.size()).to(device) - output_tensor)

    return torch.mean(loss_tensor) * torch.tensor(-1).to(device)


def dice_loss(pred, label):
    return 2*torch.sum(pred*label)/(torch.sum(pred*pred)+torch.sum(label*label))


def train_model(data, valid_data, test_data):
    model = UNet().to(device)

    opt = torch.optim.Adadelta(model.parameters())

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    current_milli_time = lambda: int(round(time.time() * 1000))

    before_time = current_milli_time()

    print("Beginning Training with nf of " + str(nf) + ".")
    print("")

    if not os.path.isdir(folder + "/control_u_net_image_output"):
        os.mkdir(folder + "/control_u_net_image_output")
    if not os.path.isdir(folder + "/during_training_models"):
        os.mkdir(folder + "/during_training_models")

    f = open(folder + "/during_training_performance.txt", "a")

    for epoch in range(0, NUM_EPOCHS):
        batch_loss = 0
        e_dice_l = 0
        epoch_before_time = current_milli_time()

        for batch in (range(0, NUM_BATCHES)):
            input_tensor = data[0][batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE].to(device)
            label_tensor = data[1][batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE].to(device)
            label_tensor = torch.clamp(torch.ceil(label_tensor.float()), 0, 1).float().to(device).view(BATCH_SIZE, 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1])

            opt.zero_grad()
            output_tensor = model(input_tensor.float())
            if batch == 0:
                if not os.path.isdir(folder + "/control_u_net_image_output/epoch_" + str(epoch)):
                    os.mkdir(folder + "/control_u_net_image_output/epoch_" + str(epoch))
                save_image(output_tensor[0, 0, :, :], folder + "/control_u_net_image_output/epoch_" + str(epoch) + "/output.png")
                save_image(input_tensor[0, :, :], folder + "/control_u_net_image_output/epoch_" + str(epoch) + "/input.png")

            train_loss = pixel_BCE(output_tensor, label_tensor.float())
            dice_l = dice_loss(output_tensor, label_tensor.float())
            train_loss.backward()
            opt.step()
            batch_loss += train_loss.item()
            e_dice_l += dice_l.item()
            # print("Batch "+str(batch+1)+" Loss : "+str(train_loss_item)+" Took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")

        opt.zero_grad()
        with torch.no_grad():
            model.eval()


            valid_loss = 0
            v_dice_l = 0
            for i in range(0, VALIDATION_NUM_BATCHES):
                input_tensor = valid_data[0][i * VALIDATION_BATCH_SIZE:(i + 1) * VALIDATION_BATCH_SIZE].to(device)
                label_tensor = valid_data[1][i * VALIDATION_BATCH_SIZE:(i + 1) * VALIDATION_BATCH_SIZE].to(device)
                label_tensor = torch.clamp(torch.ceil(label_tensor.float()), 0, 1).float().to(device).view(VALIDATION_BATCH_SIZE, 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1])

                output_tensor = model(input_tensor.float())
                loss = pixel_BCE(output_tensor, label_tensor.float())
                dice_l = dice_loss(output_tensor, label_tensor.float())
                valid_loss += loss.item()
                v_dice_l += dice_l.item()

            valid_loss /= VALIDATION_NUM_BATCHES
            epoch_loss = batch_loss / (NUM_BATCHES)
            v_dice_l /= VALIDATION_NUM_BATCHES
            e_dice_l /= NUM_BATCHES


            t_test_loss = 0
            t_dice_l = 0

            for i in range(0, TESTING_NUM_BATCHES):
                input_tensor = test_data[0][i * TESTING_BATCH_SIZE:(i + 1) * TESTING_BATCH_SIZE].to(device)
                label_tensor = test_data[1][i * TESTING_BATCH_SIZE:(i + 1) * TESTING_BATCH_SIZE].to(device)
                label_tensor = label_tensor.view(TESTING_BATCH_SIZE, 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1])

                output_tensor = model(input_tensor.float())
                test_loss = pixel_BCE(output_tensor, label_tensor.float())
                dice_l = dice_loss(output_tensor, label_tensor.float())
                t_test_loss += test_loss.item()
                t_dice_l += dice_l.item()

            t_test_loss /= TESTING_NUM_BATCHES
            t_dice_l /= TESTING_NUM_BATCHES


            epoch_after_time = current_milli_time()
            seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
            minutes = math.floor(seconds / 60)
            seconds = seconds % 60
            print("[%d]   Loss : %.5f   Validation Loss : %.5f   Testing Loss : %.5f   Dice Loss : %.5f   Validation Dice Loss : %.5f   Testing Dice Loss : %.5f Took %d minute(s) %d second(s)." % (epoch + 1, epoch_loss, valid_loss, t_test_loss, e_dice_l, v_dice_l, t_dice_l, minutes, seconds))
            f.write(str(epoch + 1) + " " + str(epoch_loss) + " " + str(valid_loss) + " " + str(t_test_loss) + " " + str(e_dice_l) + " " + str(v_dice_l) + " " + str(t_dice_l) + "\n")
            if epoch % 50 == 49:
                print("Writing models...")
                torch.save(model.state_dict(), folder + "/during_training_models/model_at_e" + str(epoch + 1) + ".pt")
            if epoch + 1 == NUM_EPOCHS:
                f.write("\n")


    after_time = current_milli_time()

    torch.save(model.state_dict(), folder + "/model.pt")
    print("")
    f.close()

    seconds = math.floor((after_time - before_time) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60

    print(str(NUM_EPOCHS) + " epochs took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

    return model

if __name__ == "__main__":
    print("Start!")
    current_milli_time = lambda: int(round(time.time() * 1000))
    before_time = current_milli_time()

    files = os.listdir(".")
    m = [int(f[9:]) for f in files if len(f) > 9 and f[0:9] == "unettrial"]
    if len(m) > 0:
        folder = "unettrial" + str(max(m) + 1)
    else:
        folder = "unettrial1"
    os.mkdir(folder)

    print("Created session folder " + folder)

    print("Loading data...")

    data = torch.load("LDEO_TRAIN.pt")

    valid_data = torch.load("LDEO_VALID.pt")

    test_data = torch.load("LDEO_TEST.pt")

    print(data.size(), valid_data.size(), test_data.size())

    after_time = current_milli_time()
    seconds = math.floor((after_time - before_time) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60
    print("Data loading took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

    model = train_model(data, valid_data, test_data)