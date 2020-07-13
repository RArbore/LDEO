import torch
from sklearn import metrics
import imageio
from torchvision.utils import save_image

nf = 24

kernel = 3

padding = 1

DATA_DIMENSIONS = [1000, 1000]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

with torch.no_grad():
    model = UNet().to(device)
    model.load_state_dict(torch.load("unettrial2/model.pt"))
    model.eval()

    image = imageio.imread("high_res.jpg").astype('float32')
    image_tensor = torch.from_numpy(image).view(10000, 10000, 1).float()/256.0
    image_tensor = image_tensor.permute(2, 0, 1).to(device)
    print(image_tensor.size())
    print(torch.min(image_tensor), torch.max(image_tensor))
    rows = []
    for r in range(10):
        row = []
        for c in range(10):
            output_tensor = model(image_tensor[:, 1000*r:1000*(r+1), 1000*c:1000*(c+1)].view(1, 1, 1000, 1000)).view(1, 1000, 1000)
            print(output_tensor.size())
            row.append(output_tensor)
        rows.append(torch.cat(row, dim=2))
    final = torch.cat(rows, dim=1)
    print(final.size())
    save_image(final, "high_res_pred.png")
